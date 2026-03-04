pub fn hash(s: &str) -> u32 {
    let mut h: u32 = 2166136261;
    for b in s.bytes() {
        h ^= b as u32;
        h = h.wrapping_mul(16777619);
    }
    h
}

#[inline(always)]
pub fn fast_exp(x: f32) -> f32 {
    let x = x.clamp(-87.0, 88.0);
    let a = (1 << 23) as f32 / std::f32::consts::LN_2;
    #[allow(clippy::excessive_precision)]
    let b = (1 << 23) as f32 * (127.0 - 0.043677448);
    let v = (a * x + b) as i32;
    f32::from_bits(v as u32)
}

#[derive(Debug, Clone)]
pub struct SigmoidTable {
    table: Vec<f32>,
}

const SIGMOID_TABLE_SIZE: usize = 512;
const MAX_SIGMOID: f32 = 8.0;

impl SigmoidTable {
    pub fn new() -> Self {
        let mut table = Vec::with_capacity(SIGMOID_TABLE_SIZE + 1);
        for i in 0..=SIGMOID_TABLE_SIZE {
            let x = (i as f32 * 2.0 * MAX_SIGMOID) / SIGMOID_TABLE_SIZE as f32 - MAX_SIGMOID;
            table.push(1.0 / (1.0 + (-x).exp()));
        }
        SigmoidTable { table }
    }

    pub fn sigmoid(&self, x: f32) -> f32 {
        if x < -MAX_SIGMOID {
            0.0
        } else if x > MAX_SIGMOID {
            1.0
        } else {
            let i = ((x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE as f32 / MAX_SIGMOID / 2.0) as usize;
            self.table[i]
        }
    }
}

#[derive(Debug, Clone)]
pub struct LogTable {
    table: Vec<f32>,
}

const LOG_TABLE_SIZE: usize = 512;

impl LogTable {
    pub fn new() -> Self {
        let mut table = Vec::with_capacity(LOG_TABLE_SIZE + 1);
        for i in 0..=LOG_TABLE_SIZE {
            let x = (i as f32 + 1e-5) / LOG_TABLE_SIZE as f32;
            table.push(x.ln());
        }
        LogTable { table }
    }

    pub fn log(&self, x: f32) -> f32 {
        if x > 1.0 {
            0.0
        } else {
            let i = (x * LOG_TABLE_SIZE as f32) as usize;
            self.table[i.min(LOG_TABLE_SIZE)]
        }
    }
}
