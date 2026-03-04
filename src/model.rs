use crate::matrix::Matrix;
use crate::utils::{self, LogTable, SigmoidTable};
use rand::Rng;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelType {
    Supervised,
    Skipgram,
    Cbow,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LossType {
    Softmax,
    NegativeSampling,
    HierarchicalSoftmax,
    OneVsAll,
}

#[derive(Clone)]
struct HeapEntry {
    idx: usize,
    prob: f32,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.prob == other.prob
    }
}
impl Eq for HeapEntry {}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .prob
            .partial_cmp(&self.prob)
            .unwrap_or(Ordering::Equal)
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    pub wi: Matrix,
    pub wo: Matrix,
    pub model_type: ModelType,
    pub loss_type: LossType,
    pub neg: usize,
    sigmoid_table: SigmoidTable,
    log_table: LogTable,
    negatives: Vec<usize>,
}

impl Model {
    pub fn new(
        wi: Matrix,
        wo: Matrix,
        model_type: ModelType,
        loss_type: LossType,
        neg: usize,
    ) -> Self {
        Model {
            wi,
            wo,
            model_type,
            loss_type,
            neg,
            sigmoid_table: SigmoidTable::new(),
            log_table: LogTable::new(),
            negatives: Vec::new(),
        }
    }

    pub fn init_negatives(&mut self, counts: &[u64]) {
        let mut negatives = Vec::new();
        let z: f64 = counts.iter().map(|&c| (c as f64).powf(0.5)).sum();
        for (i, &count) in counts.iter().enumerate() {
            let n = ((count as f64).powf(0.5) / z * 1e8) as usize;
            for _ in 0..n.max(1) {
                negatives.push(i);
            }
        }
        self.negatives = negatives;
    }

    #[inline(always)]
    fn get_negative(&self, target: usize, rng: &mut impl Rng) -> usize {
        let len = self.negatives.len();
        loop {
            let neg = unsafe { *self.negatives.get_unchecked(rng.gen_range(0..len)) };
            if neg != target {
                return neg;
            }
        }
    }

    #[inline]
    pub fn compute_hidden_into(&self, input: &[usize], hidden: &mut [f32]) {
        unsafe {
            std::ptr::write_bytes(hidden.as_mut_ptr(), 0, hidden.len());
        }

        let len = input.len();
        if len == 0 {
            return;
        }

        let m = self.wi.m;
        for &idx in input {
            if idx < m {
                self.wi.add_row_to(idx, hidden);
            }
        }

        if len > 1 {
            let scale = 1.0 / len as f32;
            for h in hidden.iter_mut() {
                *h *= scale;
            }
        }
    }

    pub fn compute_hidden(&self, input: &[usize]) -> Vec<f32> {
        let mut hidden = vec![0.0f32; self.wi.n];
        self.compute_hidden_into(input, &mut hidden);
        hidden
    }

    fn topk_from_scores(&self, scores: &[f32], k: usize, threshold: f32) -> Vec<(usize, f32)> {
        let n = scores.len();
        if k >= n / 2 {
            let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
            indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            indexed.truncate(k);
            indexed.retain(|&(_, p)| p >= threshold);
            indexed
        } else {
            let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(k + 1);
            for (i, &prob) in scores.iter().enumerate() {
                if prob < threshold {
                    continue;
                }
                if heap.len() < k {
                    heap.push(HeapEntry { idx: i, prob });
                } else if let Some(min) = heap.peek() {
                    if prob > min.prob {
                        heap.pop();
                        heap.push(HeapEntry { idx: i, prob });
                    }
                }
            }
            let mut result: Vec<(usize, f32)> = heap.into_iter().map(|e| (e.idx, e.prob)).collect();
            result.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            result
        }
    }

    fn softmax_scores(&self, hidden: &[f32], scores: &mut Vec<f32>) {
        let n_output = self.wo.m;
        scores.clear();
        scores.reserve(n_output);

        let mut max_score = f32::NEG_INFINITY;
        for i in 0..n_output {
            let s = self.wo.dot_row(i, hidden);
            scores.push(s);
            if s > max_score {
                max_score = s;
            }
        }

        let mut sum = 0.0f32;
        for s in scores.iter_mut() {
            *s = utils::fast_exp(*s - max_score);
            sum += *s;
        }
        let inv_sum = 1.0 / sum;
        for s in scores.iter_mut() {
            *s *= inv_sum;
        }
    }

    pub fn predict_softmax_buffered(
        &self,
        hidden: &[f32],
        k: i32,
        threshold: f32,
        scores: &mut Vec<f32>,
    ) -> Vec<(usize, f32)> {
        self.softmax_scores(hidden, scores);
        self.topk_from_scores(scores, k as usize, threshold)
    }

    pub fn softmax_loss_buffered(
        &mut self,
        hidden: &[f32],
        target: usize,
        lr: f32,
        grad: &mut [f32],
        scores: &mut [f32],
    ) -> f32 {
        let n_output = self.wo.m;
        let dim = hidden.len();
        let wo_data = self.wo.data.as_ptr();
        let wo_data_mut = self.wo.data.as_mut_ptr();
        let n = self.wo.n;

        let hidden_ptr = hidden.as_ptr();
        let mut max_score = f32::NEG_INFINITY;
        let chunks = dim / 4;
        let rem_start = chunks * 4;
        for i in 0..n_output {
            unsafe {
                let row_ptr = wo_data.add(i * n);
                let (mut s0, mut s1, mut s2, mut s3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                for c in 0..chunks {
                    let b = c * 4;
                    s0 = (*row_ptr.add(b)).mul_add(*hidden_ptr.add(b), s0);
                    s1 = (*row_ptr.add(b + 1)).mul_add(*hidden_ptr.add(b + 1), s1);
                    s2 = (*row_ptr.add(b + 2)).mul_add(*hidden_ptr.add(b + 2), s2);
                    s3 = (*row_ptr.add(b + 3)).mul_add(*hidden_ptr.add(b + 3), s3);
                }
                let mut dot = (s0 + s1) + (s2 + s3);
                for j in rem_start..dim {
                    dot = (*row_ptr.add(j)).mul_add(*hidden_ptr.add(j), dot);
                }
                *scores.get_unchecked_mut(i) = dot;
                if dot > max_score {
                    max_score = dot;
                }
            }
        }

        let mut sum = 0.0f32;
        for i in 0..n_output {
            unsafe {
                let s = scores.get_unchecked_mut(i);
                *s = utils::fast_exp(*s - max_score);
                sum += *s;
            }
        }
        let inv_sum = 1.0 / sum;
        for i in 0..n_output {
            unsafe {
                *scores.get_unchecked_mut(i) *= inv_sum;
            }
        }

        let loss = -self
            .log_table
            .log(unsafe { *scores.get_unchecked(target) }.max(1e-10));

        for i in 0..n_output {
            let score_i = unsafe { *scores.get_unchecked(i) };
            let alpha = if i == target {
                lr * (1.0 - score_i)
            } else {
                lr * -score_i
            };

            unsafe {
                let row_ptr = wo_data_mut.add(i * n);
                let grad_ptr = grad.as_mut_ptr();
                for j in 0..dim {
                    let wo_val = *row_ptr.add(j);
                    *grad_ptr.add(j) = alpha.mul_add(wo_val, *grad_ptr.add(j));
                    *row_ptr.add(j) = alpha.mul_add(*hidden_ptr.add(j), wo_val);
                }
            }
        }

        loss
    }

    pub fn negative_sampling_loss(
        &mut self,
        hidden: &[f32],
        target: usize,
        lr: f32,
        grad: &mut [f32],
        rng: &mut impl Rng,
    ) -> f32 {
        let mut loss = 0.0f32;
        let dim = grad.len();

        let dot = self.wo.dot_row(target, hidden);
        let score = self.sigmoid_table.sigmoid(dot);
        loss -= self.log_table.log(score.max(1e-10));
        let alpha = lr * (1.0 - score);
        let row = self.wo.row(target);
        for j in 0..dim {
            unsafe {
                *grad.get_unchecked_mut(j) += alpha * *row.get_unchecked(j);
            }
        }
        self.wo.add_to_row(target, hidden, alpha);

        for _ in 0..self.neg {
            let neg_target = self.get_negative(target, rng);
            let dot = self.wo.dot_row(neg_target, hidden);
            let score = self.sigmoid_table.sigmoid(dot);
            let alpha = -lr * score;
            loss -= self.log_table.log((1.0 - score).max(1e-10));
            let row = self.wo.row(neg_target);
            for j in 0..dim {
                unsafe {
                    *grad.get_unchecked_mut(j) += alpha * *row.get_unchecked(j);
                }
            }
            self.wo.add_to_row(neg_target, hidden, alpha);
        }

        loss
    }

    pub fn ova_loss(
        &mut self,
        hidden: &[f32],
        targets: &[usize],
        lr: f32,
        grad: &mut [f32],
    ) -> f32 {
        let n_output = self.wo.m;
        let mut loss = 0.0f32;
        let dim = grad.len();

        for i in 0..n_output {
            let dot = self.wo.dot_row(i, hidden);
            let is_target = targets.contains(&i);
            let score = self.sigmoid_table.sigmoid(dot);
            let label = if is_target { 1.0f32 } else { 0.0f32 };

            let alpha = lr * (label - score);
            loss -= if is_target {
                self.log_table.log(score.max(1e-10))
            } else {
                self.log_table.log((1.0 - score).max(1e-10))
            };

            let row = self.wo.row(i);
            for j in 0..dim {
                unsafe {
                    *grad.get_unchecked_mut(j) += alpha * *row.get_unchecked(j);
                }
            }
            self.wo.add_to_row(i, hidden, alpha);
        }

        loss
    }

    #[inline]
    pub fn update_input(&mut self, input: &[usize], grad: &[f32], scale: f32) {
        for &idx in input {
            if idx < self.wi.m {
                self.wi.add_to_row(idx, grad, scale);
            }
        }
    }
}
