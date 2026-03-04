use pyo3::prelude::*;
use pyo3::types::PyDict;

mod args;
mod dictionary;
mod fasttext;
mod matrix;
mod model;
mod utils;

use crate::fasttext::FastText;

#[pyclass]
#[derive(Clone)]
pub struct FastTextModel {
    inner: FastText,
}

#[pymethods]
impl FastTextModel {
    #[pyo3(signature = (text, k=1, threshold=0.0))]
    fn predict(&self, text: &Bound<'_, PyAny>, k: i32, threshold: f32) -> PyResult<PyObject> {
        let py = text.py();

        if let Ok(list) = text.downcast::<pyo3::types::PyList>() {
            let mut results: Vec<(Vec<String>, Vec<f32>)> = Vec::with_capacity(list.len());
            for item in list.iter() {
                let s: String = item.extract()?;
                let predictions = self.inner.predict(&s, k, threshold);
                let labels: Vec<String> = predictions.iter().map(|(l, _)| l.clone()).collect();
                let probs: Vec<f32> = predictions.iter().map(|(_, p)| *p).collect();
                results.push((labels, probs));
            }
            Ok(results.into_pyobject(py)?.into_any().unbind())
        } else {
            let s: String = text.extract()?;
            let predictions = self.inner.predict(&s, k, threshold);
            let labels: Vec<String> = predictions.iter().map(|(l, _)| l.clone()).collect();
            let probs: Vec<f32> = predictions.iter().map(|(_, p)| *p).collect();
            Ok((labels, probs).into_pyobject(py)?.into_any().unbind())
        }
    }

    #[pyo3(signature = (path, k=1))]
    fn test(&self, path: &str, k: i32) -> PyResult<(usize, f32, f32)> {
        self.inner
            .test(path, k)
            .map_err(pyo3::exceptions::PyIOError::new_err)
    }

    #[pyo3(signature = (path, k=1))]
    fn test_label<'py>(&self, path: &str, k: i32, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let results = self
            .inner
            .test_label(path, k)
            .map_err(pyo3::exceptions::PyIOError::new_err)?;
        let dict = PyDict::new(py);
        for (label, (precision, recall, count)) in &results {
            let inner = PyDict::new(py);
            inner.set_item("precision", precision)?;
            inner.set_item("recall", recall)?;
            inner.set_item("count", count)?;
            dict.set_item(label, inner)?;
        }
        Ok(dict)
    }

    fn get_word_vector(&self, word: &str) -> Vec<f32> {
        self.inner.get_word_vector(word)
    }

    fn get_sentence_vector(&self, sentence: &str) -> Vec<f32> {
        self.inner.get_sentence_vector(sentence)
    }

    fn get_input_vector(&self, index: usize) -> Vec<f32> {
        self.inner.get_input_vector(index)
    }

    fn get_input_matrix(&self) -> Vec<Vec<f32>> {
        let (m, n, data) = self.inner.get_input_matrix();
        (0..m).map(|i| data[i * n..(i + 1) * n].to_vec()).collect()
    }

    fn get_output_matrix(&self) -> Vec<Vec<f32>> {
        let (m, n, data) = self.inner.get_output_matrix();
        (0..m).map(|i| data[i * n..(i + 1) * n].to_vec()).collect()
    }

    fn get_word_id(&self, word: &str) -> i64 {
        self.inner.get_word_id(word)
    }

    fn get_subword_id(&self, subword: &str) -> usize {
        self.inner.get_subword_id(subword)
    }

    fn get_subwords(&self, word: &str) -> (Vec<String>, Vec<usize>) {
        self.inner.get_subwords(word)
    }

    fn get_line(&self, text: &str) -> (Vec<String>, Vec<String>) {
        self.inner.get_line(text)
    }

    fn save_model(&self, path: &str) -> PyResult<()> {
        self.inner
            .save_model(path)
            .map_err(pyo3::exceptions::PyIOError::new_err)
    }

    #[getter]
    fn words(&self) -> Vec<String> {
        self.inner.get_words()
    }
    #[getter]
    fn labels(&self) -> Vec<String> {
        self.inner.get_labels()
    }
    fn get_dimension(&self) -> usize {
        self.inner.get_dimension()
    }
    #[getter]
    fn is_quantized(&self) -> bool {
        false
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.args.dim
    }
    #[getter]
    fn lr(&self) -> f64 {
        self.inner.args.lr
    }
    #[getter]
    fn ws(&self) -> usize {
        self.inner.args.ws
    }
    #[getter]
    fn epoch(&self) -> usize {
        self.inner.args.epoch
    }
    #[getter]
    fn min_count(&self) -> u32 {
        self.inner.args.min_count
    }
    #[getter]
    fn min_count_label(&self) -> u32 {
        self.inner.args.min_count_label
    }
    #[getter]
    fn minn(&self) -> usize {
        self.inner.args.minn
    }
    #[getter]
    fn maxn(&self) -> usize {
        self.inner.args.maxn
    }
    #[getter]
    fn neg(&self) -> usize {
        self.inner.args.neg
    }
    #[getter]
    fn word_ngrams(&self) -> usize {
        self.inner.args.word_ngrams
    }
    #[getter]
    fn loss(&self) -> String {
        self.inner.args.loss.clone()
    }
    #[getter]
    fn bucket(&self) -> usize {
        self.inner.args.bucket
    }
    #[getter]
    fn thread(&self) -> usize {
        self.inner.args.thread
    }
    #[getter]
    fn lr_update_rate(&self) -> usize {
        self.inner.args.lr_update_rate
    }
    #[getter]
    fn t(&self) -> f64 {
        self.inner.args.t
    }
    #[getter]
    fn label(&self) -> String {
        self.inner.args.label_prefix.clone()
    }
    #[getter]
    fn verbose(&self) -> u32 {
        self.inner.args.verbose
    }

    fn __getitem__(&self, word: String) -> Vec<f32> {
        self.inner.get_word_vector(&word)
    }

    fn __contains__(&self, word: String) -> bool {
        self.inner.contains_word(&word)
    }

    fn __repr__(&self) -> String {
        format!(
            "<FastTextModel dim={} words={} labels={}>",
            self.inner.get_dimension(),
            self.inner.get_words().len(),
            self.inner.get_labels().len(),
        )
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    input, lr=0.1, dim=100, ws=5, epoch=5, min_count=1, min_count_label=0,
    minn=0, maxn=0, neg=5, word_ngrams=1, loss="softmax", bucket=2000000,
    thread=1, lr_update_rate=100, t=1e-4, label="__label__", verbose=2,
))]
fn train_supervised(
    input: &str,
    lr: f64,
    dim: usize,
    ws: usize,
    epoch: usize,
    min_count: u32,
    min_count_label: u32,
    minn: usize,
    maxn: usize,
    neg: usize,
    word_ngrams: usize,
    loss: &str,
    bucket: usize,
    thread: usize,
    lr_update_rate: usize,
    t: f64,
    label: &str,
    verbose: u32,
) -> PyResult<FastTextModel> {
    match loss {
        "softmax" | "ns" | "hs" | "ova" => {}
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown loss type: {}. Use 'softmax', 'ns', 'hs', or 'ova'",
                loss
            )))
        }
    }

    let args = args::Args {
        input: input.to_string(),
        lr,
        dim,
        ws,
        epoch,
        min_count,
        min_count_label,
        minn,
        maxn,
        neg,
        word_ngrams,
        loss: loss.to_string(),
        bucket,
        thread,
        lr_update_rate,
        t,
        verbose,
        model: "sup".to_string(),
        label_prefix: label.to_string(),
    };

    let ft = FastText::train(&args).map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    Ok(FastTextModel { inner: ft })
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    input, model="skipgram", lr=0.05, dim=100, ws=5, epoch=5, min_count=5,
    min_count_label=0, minn=3, maxn=6, neg=5, word_ngrams=1, loss="ns",
    bucket=2000000, thread=1, lr_update_rate=100, t=1e-4, label="__label__", verbose=2,
))]
fn train_unsupervised(
    input: &str,
    model: &str,
    lr: f64,
    dim: usize,
    ws: usize,
    epoch: usize,
    min_count: u32,
    min_count_label: u32,
    minn: usize,
    maxn: usize,
    neg: usize,
    word_ngrams: usize,
    loss: &str,
    bucket: usize,
    thread: usize,
    lr_update_rate: usize,
    t: f64,
    label: &str,
    verbose: u32,
) -> PyResult<FastTextModel> {
    match model {
        "skipgram" | "cbow" => {}
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown model type: {}. Use 'skipgram' or 'cbow'",
                model
            )))
        }
    }

    let args = args::Args {
        input: input.to_string(),
        lr,
        dim,
        ws,
        epoch,
        min_count,
        min_count_label,
        minn,
        maxn,
        neg,
        word_ngrams,
        loss: loss.to_string(),
        bucket,
        thread,
        lr_update_rate,
        t,
        verbose,
        model: model.to_string(),
        label_prefix: label.to_string(),
    };

    let ft = FastText::train(&args).map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    Ok(FastTextModel { inner: ft })
}

#[pyfunction]
fn load_model(path: &str) -> PyResult<FastTextModel> {
    let ft = FastText::load_model(path).map_err(pyo3::exceptions::PyIOError::new_err)?;
    Ok(FastTextModel { inner: ft })
}

#[pyfunction]
fn tokenize(text: &str) -> Vec<String> {
    FastText::tokenize(text)
}

#[pymodule]
fn fasttext_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FastTextModel>()?;
    m.add_function(wrap_pyfunction!(train_supervised, m)?)?;
    m.add_function(wrap_pyfunction!(train_unsupervised, m)?)?;
    m.add_function(wrap_pyfunction!(load_model, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize, m)?)?;
    Ok(())
}
