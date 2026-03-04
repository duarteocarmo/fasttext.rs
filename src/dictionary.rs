use crate::args::Args;
use crate::utils;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EntryType {
    Word,
    Label,
}

#[derive(Debug, Clone)]
pub struct Entry {
    pub word: String,
    pub count: u64,
    pub entry_type: EntryType,
    pub subwords: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct Dictionary {
    entries: Vec<Entry>,
    word2int: HashMap<String, usize>,
    words: Vec<usize>,
    labels: Vec<usize>,
    pub nwords: usize,
    pub nlabels: usize,
    pub ntokens: u64,
    pub min_count: u32,
    pub min_count_label: u32,
    pub bucket: usize,
    pub minn: usize,
    pub maxn: usize,
    pub word_ngrams: usize,
    pub label_prefix: String,
}

impl Dictionary {
    pub fn new(args: &Args) -> Self {
        Dictionary {
            entries: Vec::new(),
            word2int: HashMap::new(),
            words: Vec::new(),
            labels: Vec::new(),
            nwords: 0,
            nlabels: 0,
            ntokens: 0,
            min_count: args.min_count,
            min_count_label: args.min_count_label,
            bucket: args.bucket,
            minn: args.minn,
            maxn: args.maxn,
            word_ngrams: args.word_ngrams,
            label_prefix: args.label_prefix.clone(),
        }
    }

    pub fn read_from_file(&mut self, filename: &str) -> Result<(), String> {
        let file = File::open(filename).map_err(|e| format!("Cannot open {}: {}", filename, e))?;
        let reader = BufReader::new(file);

        let mut word_counts: HashMap<String, u64> = HashMap::new();
        let mut ntokens: u64 = 0;

        for line in reader.lines() {
            let line = line.map_err(|e| format!("Read error: {}", e))?;
            for token in line.split_whitespace() {
                *word_counts.entry(token.to_string()).or_insert(0) += 1;
                ntokens += 1;
            }
        }

        self.ntokens = ntokens;

        let mut entries: Vec<(String, u64)> = word_counts.into_iter().collect();
        entries.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

        for (word, count) in &entries {
            let entry_type = if word.starts_with(&self.label_prefix) {
                EntryType::Label
            } else {
                EntryType::Word
            };

            let min_c = match entry_type {
                EntryType::Word => self.min_count as u64,
                EntryType::Label => self.min_count_label as u64,
            };

            if *count >= min_c {
                let idx = self.entries.len();
                self.word2int.insert(word.clone(), idx);
                self.entries.push(Entry {
                    word: word.clone(),
                    count: *count,
                    entry_type,
                    subwords: Vec::new(),
                });

                match entry_type {
                    EntryType::Word => {
                        self.words.push(idx);
                        self.nwords += 1;
                    }
                    EntryType::Label => {
                        self.labels.push(idx);
                        self.nlabels += 1;
                    }
                }
            }
        }

        self.init_subwords();
        Ok(())
    }

    fn init_subwords(&mut self) {
        for i in 0..self.entries.len() {
            if self.entries[i].entry_type == EntryType::Word {
                let word = &self.entries[i].word;
                let mut subwords = vec![i];
                if self.maxn > 0 {
                    subwords.extend(self.compute_subwords(word));
                }
                self.entries[i].subwords = subwords;
            }
        }
    }

    pub fn compute_subwords(&self, word: &str) -> Vec<usize> {
        let mut subwords = Vec::new();
        let bounded = format!("<{}>", word);
        let chars: Vec<char> = bounded.chars().collect();
        for i in 0..chars.len() {
            let mut ngram = String::new();
            for (j, &ch) in chars.iter().enumerate().skip(i) {
                ngram.push(ch);
                let len = j - i + 1;
                if len >= self.minn && len <= self.maxn {
                    let h = utils::hash(&ngram) as usize % self.bucket;
                    subwords.push(self.nwords + h);
                }
            }
        }
        subwords
    }

    pub fn get_id(&self, word: &str) -> Option<usize> {
        self.word2int.get(word).copied()
    }

    pub fn get_entry(&self, id: usize) -> &Entry {
        &self.entries[id]
    }

    pub fn get_words(&self) -> Vec<String> {
        self.words
            .iter()
            .map(|&i| self.entries[i].word.clone())
            .collect()
    }

    pub fn get_labels(&self) -> Vec<String> {
        self.labels
            .iter()
            .map(|&i| self.entries[i].word.clone())
            .collect()
    }

    pub fn get_label_id(&self, label: &str) -> Option<usize> {
        self.labels
            .iter()
            .position(|&i| self.entries[i].word == label)
    }

    pub fn get_line_flat(&self, text: &str, word_buf: &mut Vec<usize>, label_buf: &mut Vec<usize>) {
        word_buf.clear();
        label_buf.clear();

        let label_prefix = self.label_prefix.as_bytes();
        let mut word_ids_for_ngrams: Vec<usize> = Vec::new();

        for token in text.split_ascii_whitespace() {
            if token.as_bytes().starts_with(label_prefix) {
                if let Some(lid) = self.get_label_id(token) {
                    label_buf.push(lid);
                }
            } else if let Some(&id) = self.word2int.get(token) {
                word_buf.extend_from_slice(&self.entries[id].subwords);
                word_ids_for_ngrams.push(id);
            }
        }

        if self.word_ngrams > 1 && !word_ids_for_ngrams.is_empty() {
            for i in 0..word_ids_for_ngrams.len() {
                let mut h = word_ids_for_ngrams[i] as u64;
                for &wid in &word_ids_for_ngrams
                    [(i + 1)..std::cmp::min(i + self.word_ngrams, word_ids_for_ngrams.len())]
                {
                    h = h.wrapping_mul(116049371).wrapping_add(wid as u64);
                    word_buf.push((h as usize) % self.bucket + self.nwords);
                }
            }
        }
    }

    pub fn get_subwords(&self, word: &str) -> Vec<usize> {
        if let Some(&id) = self.word2int.get(word) {
            self.entries[id].subwords.clone()
        } else {
            self.compute_subwords(word)
        }
    }

    pub fn size(&self) -> usize {
        self.entries.len()
    }

    pub fn add_entry(&mut self, word: String, count: u64, entry_type: EntryType) {
        let idx = self.entries.len();
        self.word2int.insert(word.clone(), idx);
        self.entries.push(Entry {
            word,
            count,
            entry_type,
            subwords: Vec::new(),
        });
        match entry_type {
            EntryType::Word => self.words.push(idx),
            EntryType::Label => self.labels.push(idx),
        }
    }

    pub fn reinit_subwords(&mut self) {
        self.init_subwords();
    }
}
