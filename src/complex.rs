use cubecl::{CubeType, cube};

#[derive(CubeType, Clone, Copy)]
pub struct C32 {
    pub re: f32,
    pub im: f32,
}

#[cube]
impl C32 {
    #[inline]
    pub fn add(self, b: Self) -> Self {
        C32 {
            re: self.re + b.re,
            im: self.im + b.im,
        }
    }
    
    #[inline]
    pub fn sub(self, b: Self) -> Self {
        C32 {
            re: self.re - b.re,
            im: self.im - b.im,
        }
    }

    #[inline]
    pub fn mul(self, b: Self) -> Self {
        C32 {
            re: self.re * b.re - self.im * b.im,
            im: self.re * b.im + self.im * b.re,
        }
    }

    #[inline]
    pub fn scale(self, s: f32) -> Self {
        C32 {
            re: self.re * s,
            im: self.im * s,
        }
    }
}
