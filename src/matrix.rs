use rand::Rng;
use rand::SeedableRng;

#[derive(Debug, Clone)]
pub struct Matrix {
    pub m: usize,
    pub n: usize,
    pub data: Vec<f32>,
}

impl Matrix {
    pub fn new(m: usize, n: usize) -> Self {
        Matrix {
            m,
            n,
            data: vec![0.0; m * n],
        }
    }

    pub fn random(m: usize, n: usize, seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let bound = 1.0 / n as f32;
        let data: Vec<f32> = (0..m * n).map(|_| rng.gen_range(-bound..bound)).collect();
        Matrix { m, n, data }
    }

    #[inline(always)]
    pub fn row(&self, i: usize) -> &[f32] {
        let start = i * self.n;
        unsafe { self.data.get_unchecked(start..start + self.n) }
    }

    #[inline(always)]
    pub fn add_to_row(&mut self, i: usize, vec: &[f32], alpha: f32) {
        let start = i * self.n;
        let n = self.n;
        unsafe {
            let row_ptr = self.data.as_mut_ptr().add(start);
            let vec_ptr = vec.as_ptr();
            for j in 0..n {
                let r = row_ptr.add(j);
                *r = alpha.mul_add(*vec_ptr.add(j), *r);
            }
        }
    }

    #[inline(always)]
    pub fn dot_row(&self, i: usize, vec: &[f32]) -> f32 {
        let start = i * self.n;
        let n = self.n;
        unsafe {
            let row_ptr = self.data.as_ptr().add(start);
            let vec_ptr = vec.as_ptr();
            let mut s0 = 0.0f32;
            let mut s1 = 0.0f32;
            let mut s2 = 0.0f32;
            let mut s3 = 0.0f32;
            let chunks = n / 4;
            for c in 0..chunks {
                let b = c * 4;
                s0 = (*row_ptr.add(b)).mul_add(*vec_ptr.add(b), s0);
                s1 = (*row_ptr.add(b + 1)).mul_add(*vec_ptr.add(b + 1), s1);
                s2 = (*row_ptr.add(b + 2)).mul_add(*vec_ptr.add(b + 2), s2);
                s3 = (*row_ptr.add(b + 3)).mul_add(*vec_ptr.add(b + 3), s3);
            }
            let mut sum = (s0 + s1) + (s2 + s3);
            for j in (chunks * 4)..n {
                sum = (*row_ptr.add(j)).mul_add(*vec_ptr.add(j), sum);
            }
            sum
        }
    }

    #[inline(always)]
    pub fn add_row_to(&self, i: usize, dst: &mut [f32]) {
        let start = i * self.n;
        let n = self.n;
        unsafe {
            let row_ptr = self.data.as_ptr().add(start);
            let dst_ptr = dst.as_mut_ptr();
            for j in 0..n {
                *dst_ptr.add(j) += *row_ptr.add(j);
            }
        }
    }
}
