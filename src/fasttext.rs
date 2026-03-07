use crate::args::Args;
use crate::dictionary::Dictionary;
use crate::matrix::Matrix;
use crate::model::{LossType, Model, ModelType};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use rand::Rng;
use rand::SeedableRng;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};

const FASTTEXT_FILEFORMAT_MAGIC_INT32: i32 = 793712314;
const FASTTEXT_VERSION: i32 = 12;

struct PrecomputedData {
    all_words: Vec<usize>,
    all_labels: Vec<usize>,
    line_info: Vec<(u32, u32, u32, u32, u32)>,
}

impl PrecomputedData {
    fn words(&self, i: usize) -> &[usize] {
        let (start, len, _, _, _) = self.line_info[i];
        &self.all_words[start as usize..(start + len) as usize]
    }

    fn labels(&self, i: usize) -> &[usize] {
        let (_, _, start, len, _) = self.line_info[i];
        &self.all_labels[start as usize..(start + len) as usize]
    }

    fn ntokens(&self, i: usize) -> u32 {
        self.line_info[i].4
    }

    fn num_lines(&self) -> usize {
        self.line_info.len()
    }
}

#[derive(Debug, Clone)]
pub struct FastText {
    pub args: Args,
    pub dict: Dictionary,
    pub model: Model,
    cached_labels: Vec<String>,
}

impl FastText {
    pub fn train(args: &Args) -> Result<Self, String> {
        let mut dict = Dictionary::new(args);
        dict.read_from_file(&args.input)?;

        if args.verbose > 0 {
            eprintln!(
                "Read {} tokens, {} words, {} labels",
                dict.ntokens, dict.nwords, dict.nlabels
            );
        }

        let model_type = match args.model.as_str() {
            "sup" => ModelType::Supervised,
            "skipgram" => ModelType::Skipgram,
            "cbow" => ModelType::Cbow,
            _ => return Err(format!("Unknown model type: {}", args.model)),
        };

        let loss_type = match args.loss.as_str() {
            "softmax" => LossType::Softmax,
            "ns" => LossType::NegativeSampling,
            "hs" => LossType::HierarchicalSoftmax,
            "ova" => LossType::OneVsAll,
            _ => return Err(format!("Unknown loss type: {}", args.loss)),
        };

        let n_input = dict.nwords + dict.bucket;
        let wi = Matrix::random(n_input, args.dim, 42);

        let n_output = match model_type {
            ModelType::Supervised => dict.nlabels,
            _ => dict.nwords,
        };
        let wo = Matrix::new(n_output, args.dim);

        let mut model = Model::new(wi, wo, model_type, loss_type, args.neg);

        if loss_type == LossType::NegativeSampling {
            let counts: Vec<u64> = match model_type {
                ModelType::Supervised => (0..dict.nlabels)
                    .map(|i| {
                        let labels = dict.get_labels();
                        dict.get_entry(dict.get_id(&labels[i]).unwrap()).count
                    })
                    .collect(),
                _ => {
                    let words = dict.get_words();
                    words
                        .iter()
                        .map(|w| dict.get_entry(dict.get_id(w).unwrap()).count)
                        .collect()
                }
            };
            model.init_negatives(&counts);
        }

        let cached_labels = dict.get_labels();
        let mut ft = FastText {
            args: args.clone(),
            dict,
            model,
            cached_labels,
        };
        ft.train_model()?;
        Ok(ft)
    }

    fn train_model(&mut self) -> Result<(), String> {
        let file = File::open(&self.args.input)
            .map_err(|e| format!("Cannot open {}: {}", self.args.input, e))?;
        let reader = BufReader::new(file);
        let lines: Vec<String> = reader.lines().map_while(Result::ok).collect();

        if lines.is_empty() {
            return Err("Empty input file".to_string());
        }

        let mut all_words = Vec::with_capacity(lines.len() * 20);
        let mut all_labels = Vec::with_capacity(lines.len() * 2);
        let mut line_info = Vec::with_capacity(lines.len());
        let mut word_buf = Vec::new();
        let mut label_buf = Vec::new();

        for line in &lines {
            self.dict.get_line_flat(line, &mut word_buf, &mut label_buf);
            let ntokens = line.split_ascii_whitespace().count() as u32;
            let ws = all_words.len() as u32;
            let wl = word_buf.len() as u32;
            let ls = all_labels.len() as u32;
            let ll = label_buf.len() as u32;
            all_words.extend_from_slice(&word_buf);
            all_labels.extend_from_slice(&label_buf);
            line_info.push((ws, wl, ls, ll, ntokens));
        }
        drop(lines);

        let precomputed = PrecomputedData {
            all_words,
            all_labels,
            line_info,
        };

        let total_tokens = self.dict.ntokens as f64 * self.args.epoch as f64;
        let mut tokens_processed: u64 = 0;
        let mut loss_sum = 0.0f32;
        let mut loss_count = 0u64;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let dim = self.args.dim;
        let n_output = self.model.wo.m;
        let mut hidden = vec![0.0f32; dim];
        let mut grad = vec![0.0f32; dim];
        let mut scores = vec![0.0f32; n_output];

        let num_lines = precomputed.num_lines();
        for epoch in 0..self.args.epoch {
            for line_idx in 0..num_lines {
                let progress = tokens_processed as f64 / total_tokens;
                let lr = (self.args.lr * (1.0 - progress)).max(0.0) as f32;
                tokens_processed += precomputed.ntokens(line_idx) as u64;

                match self.model.model_type {
                    ModelType::Supervised => {
                        let words = precomputed.words(line_idx);
                        let labels = precomputed.labels(line_idx);
                        if words.is_empty() || labels.is_empty() {
                            continue;
                        }
                        let l = self.train_supervised_precomputed(
                            words,
                            labels,
                            lr,
                            &mut hidden,
                            &mut grad,
                            &mut scores,
                            &mut rng,
                        );
                        if l > 0.0 {
                            loss_sum += l;
                            loss_count += 1;
                        }
                    }
                    ModelType::Skipgram | ModelType::Cbow => {
                        loss_count += 1;
                    }
                }
            }

            if self.args.verbose > 0 {
                let avg_loss = if loss_count > 0 {
                    loss_sum / loss_count as f32
                } else {
                    0.0
                };
                eprintln!(
                    "Epoch {}/{} - lr: {:.6} - loss: {:.6} - progress: {:.1}%",
                    epoch + 1,
                    self.args.epoch,
                    self.args.lr * (1.0 - (tokens_processed as f64 / total_tokens)),
                    avg_loss,
                    (epoch + 1) as f64 / self.args.epoch as f64 * 100.0,
                );
            }
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn train_supervised_precomputed(
        &mut self,
        input: &[usize],
        labels: &[usize],
        lr: f32,
        hidden: &mut [f32],
        grad: &mut [f32],
        scores: &mut [f32],
        rng: &mut impl Rng,
    ) -> f32 {
        self.model.compute_hidden_into(input, hidden);
        for g in grad.iter_mut() {
            *g = 0.0;
        }

        let loss = match self.model.loss_type {
            LossType::Softmax | LossType::HierarchicalSoftmax => {
                let target = labels[rng.gen_range(0..labels.len())];
                self.model
                    .softmax_loss_buffered(hidden, target, lr, grad, scores)
            }
            LossType::NegativeSampling => {
                let target = labels[rng.gen_range(0..labels.len())];
                self.model
                    .negative_sampling_loss(hidden, target, lr, grad, rng)
            }
            LossType::OneVsAll => self.model.ova_loss(hidden, labels, lr, grad),
        };

        let scale = 1.0 / input.len() as f32;
        self.model.update_input(input, grad, scale);
        loss
    }

    pub fn predict(&self, text: &str, k: i32, threshold: f32) -> Vec<(String, f32)> {
        thread_local! {
            static WORD_BUF: std::cell::RefCell<Vec<usize>> = std::cell::RefCell::new(Vec::with_capacity(256));
            static LABEL_BUF: std::cell::RefCell<Vec<usize>> = std::cell::RefCell::new(Vec::with_capacity(16));
            static HIDDEN_BUF: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
            static SCORES_BUF: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
        }

        WORD_BUF.with(|wb| {
            LABEL_BUF.with(|lb| {
                HIDDEN_BUF.with(|hb| {
                    SCORES_BUF.with(|sb| {
                        let mut word_buf = wb.borrow_mut();
                        let mut label_buf = lb.borrow_mut();
                        let mut hidden = hb.borrow_mut();
                        let mut scores = sb.borrow_mut();

                        self.dict.get_line_flat(text, &mut word_buf, &mut label_buf);
                        if word_buf.is_empty() {
                            return Vec::new();
                        }

                        let dim = self.args.dim;
                        hidden.resize(dim, 0.0);
                        self.model.compute_hidden_into(&word_buf, &mut hidden);

                        let labels = &self.cached_labels;
                        match self.model.loss_type {
                            LossType::OneVsAll => {
                                let n_output = self.model.wo.m;
                                let mut indexed: Vec<(usize, f32)> = (0..n_output)
                                    .map(|i| {
                                        let dot = self.model.wo.dot_row(i, &hidden);
                                        (i, 1.0 / (1.0 + (-dot).exp()))
                                    })
                                    .collect();
                                indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                                indexed.truncate(k as usize);
                                indexed.retain(|&(_, p)| p >= threshold);
                                indexed
                                    .into_iter()
                                    .map(|(i, p)| (labels[i].clone(), p))
                                    .collect()
                            }
                            _ => {
                                let preds = self.model.predict_softmax_buffered(
                                    &hidden,
                                    k,
                                    threshold,
                                    &mut scores,
                                );
                                preds
                                    .into_iter()
                                    .map(|(i, p)| (labels[i].clone(), p))
                                    .collect()
                            }
                        }
                    })
                })
            })
        })
    }

    pub fn get_word_vector(&self, word: &str) -> Vec<f32> {
        let subwords = self.dict.get_subwords(word);
        self.model.compute_hidden(&subwords)
    }

    pub fn get_sentence_vector(&self, sentence: &str) -> Vec<f32> {
        let dim = self.args.dim;
        let mut vec = vec![0.0f32; dim];
        let mut count = 0;

        for word in sentence.split_whitespace() {
            if !word.starts_with(&self.dict.label_prefix) {
                let wv = self.get_word_vector(word);
                for (v, w) in vec.iter_mut().zip(wv.iter()) {
                    *v += w;
                }
                count += 1;
            }
        }

        if count > 0 {
            let scale = 1.0 / count as f32;
            for v in vec.iter_mut() {
                *v *= scale;
            }
        }

        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in vec.iter_mut() {
                *v /= norm;
            }
        }

        vec
    }

    pub fn get_words(&self) -> Vec<String> {
        self.dict.get_words()
    }
    pub fn get_labels(&self) -> Vec<String> {
        self.dict.get_labels()
    }
    pub fn get_dimension(&self) -> usize {
        self.args.dim
    }

    pub fn get_word_id(&self, word: &str) -> i64 {
        self.dict.get_id(word).map(|id| id as i64).unwrap_or(-1)
    }

    pub fn get_subword_id(&self, subword: &str) -> usize {
        if self.dict.bucket == 0 {
            return 0;
        }
        let h = crate::utils::hash(subword) as usize % self.dict.bucket;
        self.dict.nwords + h
    }

    pub fn get_subwords(&self, word: &str) -> (Vec<String>, Vec<usize>) {
        let indices = self.dict.get_subwords(word);
        let bounded = format!("<{}>", word);
        let chars: Vec<char> = bounded.chars().collect();
        let mut subword_strings = Vec::new();

        if self.dict.get_id(word).is_some() {
            subword_strings.push(word.to_string());
        }

        if self.dict.maxn > 0 {
            for i in 0..chars.len() {
                let mut ngram = String::new();
                for (j, &ch) in chars.iter().enumerate().skip(i) {
                    ngram.push(ch);
                    let len = j - i + 1;
                    if len >= self.dict.minn && len <= self.dict.maxn {
                        subword_strings.push(ngram.clone());
                    }
                }
            }
        }

        (subword_strings, indices)
    }

    pub fn get_line(&self, text: &str) -> (Vec<String>, Vec<String>) {
        let words: Vec<String> = text
            .split_ascii_whitespace()
            .filter(|t| !t.starts_with(&self.dict.label_prefix))
            .filter(|t| self.dict.get_id(t).is_some())
            .map(|s| s.to_string())
            .collect();

        let labels: Vec<String> = text
            .split_ascii_whitespace()
            .filter(|t| t.starts_with(&self.dict.label_prefix))
            .filter(|t| self.dict.get_label_id(t).is_some())
            .map(|s| s.to_string())
            .collect();

        (words, labels)
    }

    pub fn get_input_vector(&self, index: usize) -> Vec<f32> {
        if index < self.model.wi.m {
            self.model.wi.row(index).to_vec()
        } else {
            vec![0.0; self.args.dim]
        }
    }

    pub fn get_input_matrix(&self) -> (usize, usize, Vec<f32>) {
        (self.model.wi.m, self.model.wi.n, self.model.wi.data.clone())
    }

    pub fn get_output_matrix(&self) -> (usize, usize, Vec<f32>) {
        (self.model.wo.m, self.model.wo.n, self.model.wo.data.clone())
    }

    pub fn contains_word(&self, word: &str) -> bool {
        self.dict.get_id(word).is_some()
    }

    pub fn tokenize(text: &str) -> Vec<String> {
        text.split_ascii_whitespace()
            .map(|s| s.to_string())
            .collect()
    }

    pub fn test(&self, path: &str, k: i32) -> Result<(usize, f32, f32), String> {
        let file = File::open(path).map_err(|e| format!("Cannot open {}: {}", path, e))?;
        let reader = BufReader::new(file);

        let mut n = 0usize;
        let mut precision_sum = 0.0f32;
        let mut recall_sum = 0.0f32;

        for line in reader.lines() {
            let line = line.map_err(|e| format!("Read error: {}", e))?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let parts: Vec<&str> = line.split_ascii_whitespace().collect();
            let true_labels: Vec<&str> = parts
                .iter()
                .filter(|p| p.starts_with("__label__"))
                .copied()
                .collect();
            let text: String = parts
                .iter()
                .filter(|p| !p.starts_with("__label__"))
                .copied()
                .collect::<Vec<&str>>()
                .join(" ");

            if true_labels.is_empty() || text.is_empty() {
                continue;
            }

            let predictions = self.predict(&text, k, 0.0);
            let pred_labels: Vec<&str> = predictions.iter().map(|(l, _)| l.as_str()).collect();

            let correct = pred_labels
                .iter()
                .filter(|pl| true_labels.contains(pl))
                .count();
            let p = if pred_labels.is_empty() {
                0.0
            } else {
                correct as f32 / pred_labels.len() as f32
            };
            let r = if true_labels.is_empty() {
                0.0
            } else {
                correct as f32 / true_labels.len() as f32
            };

            precision_sum += p;
            recall_sum += r;
            n += 1;
        }

        if n == 0 {
            return Ok((0, 0.0, 0.0));
        }
        Ok((n, precision_sum / n as f32, recall_sum / n as f32))
    }

    pub fn test_label(
        &self,
        path: &str,
        k: i32,
    ) -> Result<std::collections::HashMap<String, (f32, f32, usize)>, String> {
        use std::collections::HashMap;

        let file = File::open(path).map_err(|e| format!("Cannot open {}: {}", path, e))?;
        let reader = BufReader::new(file);

        let mut label_stats: HashMap<String, (usize, usize, usize)> = HashMap::new();

        for line in reader.lines() {
            let line = line.map_err(|e| format!("Read error: {}", e))?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let parts: Vec<&str> = line.split_ascii_whitespace().collect();
            let true_labels: Vec<String> = parts
                .iter()
                .filter(|p| p.starts_with("__label__"))
                .map(|s| s.to_string())
                .collect();
            let text: String = parts
                .iter()
                .filter(|p| !p.starts_with("__label__"))
                .copied()
                .collect::<Vec<&str>>()
                .join(" ");

            if true_labels.is_empty() || text.is_empty() {
                continue;
            }

            let predictions = self.predict(&text, k, 0.0);
            let pred_labels: Vec<String> = predictions.into_iter().map(|(l, _)| l).collect();

            for pl in &pred_labels {
                let entry = label_stats.entry(pl.clone()).or_insert((0, 0, 0));
                if true_labels.contains(pl) {
                    entry.0 += 1;
                } else {
                    entry.1 += 1;
                }
            }

            for tl in &true_labels {
                if !pred_labels.contains(tl) {
                    label_stats.entry(tl.clone()).or_insert((0, 0, 0)).2 += 1;
                }
            }
        }

        let mut result = HashMap::new();
        for (label, (tp, fp, fneg)) in &label_stats {
            let precision = if tp + fp > 0 {
                *tp as f32 / (tp + fp) as f32
            } else {
                0.0
            };
            let recall = if tp + fneg > 0 {
                *tp as f32 / (tp + fneg) as f32
            } else {
                0.0
            };
            result.insert(label.clone(), (precision, recall, tp + fneg));
        }

        Ok(result)
    }

    pub fn save_model(&self, path: &str) -> Result<(), String> {
        let file = File::create(path).map_err(|e| format!("Cannot create {}: {}", path, e))?;
        let mut writer = BufWriter::new(file);
        let m = |e: std::io::Error| e.to_string();

        writer
            .write_i32::<LittleEndian>(FASTTEXT_FILEFORMAT_MAGIC_INT32)
            .map_err(m)?;
        writer
            .write_i32::<LittleEndian>(FASTTEXT_VERSION)
            .map_err(m)?;

        self.write_args(&mut writer)?;
        self.write_dict(&mut writer)?;
        writer.write_u8(0).map_err(m)?; // quant flag for input
        self.write_matrix(&mut writer, &self.model.wi)?;
        writer.write_u8(0).map_err(m)?; // quant flag for output
        self.write_matrix(&mut writer, &self.model.wo)?;

        Ok(())
    }

    fn write_args(&self, writer: &mut impl Write) -> Result<(), String> {
        let m = |e: std::io::Error| e.to_string();
        writer
            .write_i32::<LittleEndian>(self.args.dim as i32)
            .map_err(m)?;
        writer
            .write_i32::<LittleEndian>(self.args.ws as i32)
            .map_err(m)?;
        writer
            .write_i32::<LittleEndian>(self.args.epoch as i32)
            .map_err(m)?;
        writer
            .write_i32::<LittleEndian>(self.args.min_count as i32)
            .map_err(m)?;
        writer
            .write_i32::<LittleEndian>(self.args.neg as i32)
            .map_err(m)?;
        writer
            .write_i32::<LittleEndian>(self.args.word_ngrams as i32)
            .map_err(m)?;

        let loss_id: i32 = match self.args.loss.as_str() {
            "hs" => 1,
            "ns" => 2,
            "softmax" => 3,
            "ova" => 4,
            _ => 3,
        };
        writer.write_i32::<LittleEndian>(loss_id).map_err(m)?;

        let model_id: i32 = match self.args.model.as_str() {
            "cbow" => 1,
            "skipgram" | "sg" => 2,
            "sup" => 3,
            _ => 3,
        };
        writer.write_i32::<LittleEndian>(model_id).map_err(m)?;

        writer
            .write_i32::<LittleEndian>(self.args.bucket as i32)
            .map_err(m)?;
        writer
            .write_i32::<LittleEndian>(self.args.minn as i32)
            .map_err(m)?;
        writer
            .write_i32::<LittleEndian>(self.args.maxn as i32)
            .map_err(m)?;
        writer
            .write_i32::<LittleEndian>(self.args.lr_update_rate as i32)
            .map_err(m)?;
        writer.write_f64::<LittleEndian>(self.args.t).map_err(m)?;

        Ok(())
    }

    fn write_dict(&self, writer: &mut impl Write) -> Result<(), String> {
        let m = |e: std::io::Error| e.to_string();
        writer
            .write_i32::<LittleEndian>(self.dict.size() as i32)
            .map_err(m)?;
        writer
            .write_i32::<LittleEndian>(self.dict.nwords as i32)
            .map_err(m)?;
        writer
            .write_i32::<LittleEndian>(self.dict.nlabels as i32)
            .map_err(m)?;
        writer
            .write_i64::<LittleEndian>(self.dict.ntokens as i64)
            .map_err(m)?;
        writer.write_i64::<LittleEndian>(0).map_err(m)?; // pruneidx size

        for i in 0..self.dict.size() {
            let entry = self.dict.get_entry(i);
            writer.write_all(entry.word.as_bytes()).map_err(m)?;
            writer.write_u8(0).map_err(m)?;
            writer
                .write_i64::<LittleEndian>(entry.count as i64)
                .map_err(m)?;
            let etype: u8 = match entry.entry_type {
                crate::dictionary::EntryType::Word => 0,
                crate::dictionary::EntryType::Label => 1,
            };
            writer.write_u8(etype).map_err(m)?;
        }

        Ok(())
    }

    fn write_matrix(&self, writer: &mut impl Write, matrix: &Matrix) -> Result<(), String> {
        let m = |e: std::io::Error| e.to_string();
        writer
            .write_i64::<LittleEndian>(matrix.m as i64)
            .map_err(m)?;
        writer
            .write_i64::<LittleEndian>(matrix.n as i64)
            .map_err(m)?;
        for &val in &matrix.data {
            writer.write_f32::<LittleEndian>(val).map_err(m)?;
        }
        Ok(())
    }

    pub fn load_model(path: &str) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("Cannot open {}: {}", path, e))?;
        let mut reader = BufReader::new(file);
        let m = |e: std::io::Error| e.to_string();

        let magic = reader.read_i32::<LittleEndian>().map_err(m)?;
        let _version = reader.read_i32::<LittleEndian>().map_err(m)?;

        if magic != FASTTEXT_FILEFORMAT_MAGIC_INT32 {
            return Err("Invalid file format (bad magic number)".to_string());
        }

        let args = Self::read_args(&mut reader)?;
        let dict = Self::read_dict(&mut reader, &args)?;
        let qi = reader.read_u8().map_err(m)?;
        if qi != 0 {
            return Err("Quantized input matrices are not supported".to_string());
        }
        let wi = Self::read_matrix(&mut reader)?;

        let qo = reader.read_u8().map_err(m)?;
        if qo != 0 {
            return Err("Quantized output matrices are not supported".to_string());
        }
        let wo = Self::read_matrix(&mut reader)?;

        let model_type = match args.model.as_str() {
            "sup" => ModelType::Supervised,
            "skipgram" => ModelType::Skipgram,
            "cbow" => ModelType::Cbow,
            _ => ModelType::Supervised,
        };

        let loss_type = match args.loss.as_str() {
            "softmax" => LossType::Softmax,
            "ns" => LossType::NegativeSampling,
            "hs" => LossType::HierarchicalSoftmax,
            "ova" => LossType::OneVsAll,
            _ => LossType::Softmax,
        };

        let model = Model::new(wi, wo, model_type, loss_type, args.neg);
        let cached_labels = dict.get_labels();
        Ok(FastText {
            args,
            dict,
            model,
            cached_labels,
        })
    }

    fn read_args(reader: &mut impl Read) -> Result<Args, String> {
        let m = |e: std::io::Error| e.to_string();

        let dim = reader.read_i32::<LittleEndian>().map_err(m)? as usize;
        let ws = reader.read_i32::<LittleEndian>().map_err(m)? as usize;
        let epoch = reader.read_i32::<LittleEndian>().map_err(m)? as usize;
        let min_count = reader.read_i32::<LittleEndian>().map_err(m)? as u32;
        let neg = reader.read_i32::<LittleEndian>().map_err(m)? as usize;
        let word_ngrams = reader.read_i32::<LittleEndian>().map_err(m)? as usize;

        let loss_id = reader.read_i32::<LittleEndian>().map_err(m)?;
        let loss = match loss_id {
            1 => "hs",
            2 => "ns",
            3 => "softmax",
            4 => "ova",
            _ => "softmax",
        }
        .to_string();

        let model_id = reader.read_i32::<LittleEndian>().map_err(m)?;
        let model = match model_id {
            1 => "cbow",
            2 => "skipgram",
            3 => "sup",
            _ => "sup",
        }
        .to_string();

        let bucket = reader.read_i32::<LittleEndian>().map_err(m)? as usize;
        let minn = reader.read_i32::<LittleEndian>().map_err(m)? as usize;
        let maxn = reader.read_i32::<LittleEndian>().map_err(m)? as usize;
        let lr_update_rate = reader.read_i32::<LittleEndian>().map_err(m)? as usize;
        let t = reader.read_f64::<LittleEndian>().map_err(m)?;

        Ok(Args {
            input: String::new(),
            lr: 0.0,
            dim,
            ws,
            epoch,
            min_count,
            min_count_label: 0,
            minn,
            maxn,
            neg,
            word_ngrams,
            loss,
            bucket,
            thread: 1,
            lr_update_rate,
            t,
            verbose: 0,
            model,
            label_prefix: "__label__".to_string(),
        })
    }

    fn read_dict(reader: &mut impl Read, args: &Args) -> Result<Dictionary, String> {
        let m = |e: std::io::Error| e.to_string();

        let size = reader.read_i32::<LittleEndian>().map_err(m)? as usize;
        let nwords = reader.read_i32::<LittleEndian>().map_err(m)? as usize;
        let nlabels = reader.read_i32::<LittleEndian>().map_err(m)? as usize;
        let ntokens = reader.read_i64::<LittleEndian>().map_err(m)? as u64;
        let pruneidx_size = reader.read_i64::<LittleEndian>().map_err(m)?;

        let mut dict = Dictionary::new(args);
        dict.nwords = nwords;
        dict.nlabels = nlabels;
        dict.ntokens = ntokens;

        for _ in 0..size {
            let mut word_bytes = Vec::new();
            loop {
                let b = reader.read_u8().map_err(m)?;
                if b == 0 {
                    break;
                }
                word_bytes.push(b);
            }
            let word = String::from_utf8_lossy(&word_bytes).to_string();
            let count = reader.read_i64::<LittleEndian>().map_err(m)? as u64;
            let etype = reader.read_u8().map_err(m)?;
            let entry_type = match etype {
                0 => crate::dictionary::EntryType::Word,
                _ => crate::dictionary::EntryType::Label,
            };
            dict.add_entry(word, count, entry_type);
        }

        for _ in 0..pruneidx_size {
            reader.read_i32::<LittleEndian>().map_err(m)?;
            reader.read_i32::<LittleEndian>().map_err(m)?;
        }

        dict.reinit_subwords();
        Ok(dict)
    }

    fn read_matrix(reader: &mut impl Read) -> Result<Matrix, String> {
        let m = |e: std::io::Error| e.to_string();
        let rows = reader.read_i64::<LittleEndian>().map_err(m)? as usize;
        let cols = reader.read_i64::<LittleEndian>().map_err(m)? as usize;
        let mut data = vec![0.0f32; rows * cols];
        for val in data.iter_mut() {
            *val = reader.read_f32::<LittleEndian>().map_err(m)?;
        }
        Ok(Matrix {
            m: rows,
            n: cols,
            data,
        })
    }
}
