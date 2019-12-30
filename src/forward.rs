use num_traits::real::Real;
use std::ops;

#[derive(Debug, Clone, PartialEq, Copy)]
pub struct Dual<R: Real> {
    pub real: R,
    pub grad: R,
}

impl<R: Real> Dual<R> {
    pub fn real(self) -> R {
        self.real
    }
    pub fn grad(self) -> R {
        self.grad
    }
    pub fn abs(self) -> Self {
        Dual { real: self.real.abs(), grad: self.grad.abs() * self.real.signum() }
    }
    pub fn pow(self, n: usize) -> Self {
        if n == 1 {
            self
        } else {
            let mut result = self.clone();
            for _ in 1..n {
                result *= result.clone();
            };
            result
        }
    }
    pub fn exp(self) -> Self {
        Dual { real: self.real.exp(), grad: self.grad * self.real.exp() }
    }
    pub fn ln(self) -> Self {
        Dual { real: self.real.ln(), grad: self.grad / self.real }
    }
    pub fn sin(self) -> Self {
        Dual { real: self.real.sin(), grad: self.grad * self.real.cos() }
    }
    pub fn cos(self) -> Self {
        Dual { real: self.real.cos(), grad: -self.grad * self.real.sin() }
    }
    pub fn tan(self) -> Self {
        Dual { real: self.real.tan(), grad: self.grad / (self.real.cos() * self.real.cos())  }
    }
    pub fn asin(self) -> Self {
        Dual { real: self.real.asin(), grad: self.grad / (self.real.signum().abs() - self.real * self.real).sqrt() }
    }
    pub fn acos(self) -> Self {
        Dual { real: self.real.acos(), grad: -self.grad / (self.real.signum().abs() - self.real * self.real).sqrt() }
    }
    pub fn atan(self) -> Self {
        Dual { real: self.real.atan(), grad: self.grad / (self.real.signum().abs() + self.real * self.real) }
    }
    pub fn sinh(self) -> Self {
        Dual { real: self.real.sinh(), grad: self.grad * self.real.cosh() }
    }
    pub fn cosh(self) -> Self {
        Dual { real: self.real.cosh(), grad: self.grad * self.real.sinh() }
    }
    pub fn tanh(self) -> Self {
        Dual { real: self.real.tanh(), grad: self.grad * (self.real.exp() - (-self.real).exp()) / (self.real.exp() + (-self.real).exp()) }
    }
    pub fn asinh(self) -> Self {
        Dual { real: self.real.asinh(), grad: self.grad / (self.real.signum().abs() + self.real * self.real).sqrt() }
    }
    pub fn acosh(self) -> Self {
        Dual { real: self.real.acos(), grad: self.grad / (-self.real.signum().abs() + self.real * self.real).sqrt() }
    }
    pub fn atanh(self) -> Self {
        Dual { real: self.real.atanh(), grad: self.grad / (self.real.signum().abs() - self.real * self.real) }
    }
}

impl<R> ops::Add for Dual<R>
where
    R: Real,
{
    type Output = Dual<R>;
    fn add(self, other: Self) -> Dual<R> {
        Dual { real: self.real + other.real, grad: self.grad + other.grad }
    }
}

impl <R> ops::AddAssign for Dual<R>
where
    R :Real {
    fn add_assign(&mut self, other: Self) {
        *self = Self { real: self.real + other.real, grad: self.grad + other.grad };
    }
}

impl<R> ops::Div for Dual<R>
where
    R: Real,
{
    type Output = Dual<R>;
    fn div(self, other: Self) -> Dual<R> {
        Dual { real: self.real / other.real, grad: self.real / other.grad + self.grad / other.real }
    }
}

impl <R> ops::DivAssign for Dual<R>
where
    R :Real {
    fn div_assign(&mut self, other: Self) {
        *self = Self { real: self.real / other.real, grad: self.real / other.grad + self.grad / other.real }
    }
}

impl<R> ops::Mul for Dual<R>
where
    R: Real,
{
    type Output = Dual<R>;
    fn mul(self, other: Self) -> Dual<R> {
        Dual { real: self.real * other.real, grad: self.real * other.grad + self.grad * other.real }
    }
}

impl <R> ops::MulAssign for Dual<R>
where
    R :Real {
    fn mul_assign(&mut self, other: Self) {
        *self = Self { real: self.real * other.real, grad: self.real * other.grad + self.grad * other.real }
    }
}

impl<R> ops::Neg for Dual<R>
where
    R: Real,
{
    type Output = Dual<R>;
    fn neg(self) -> Dual<R> {
        Dual { real: -self.grad, grad: -self.real }
    }
}

impl<R> ops::Sub for Dual<R>
where
    R: Real,
{
    type Output = Dual<R>;
    fn sub(self, other: Self) -> Dual<R> {
        Dual { real: self.real - other.real, grad: self.grad - other.grad }
    }
}

impl <R> ops::SubAssign for Dual<R>
where
    R :Real {
    fn sub_assign(&mut self, other: Self) {
        *self = Self { real: self.real - other.real, grad: self.grad - other.grad };
    }
}

#[cfg(test)]
mod tests {
    use crate::forward::*;

    #[test]
    fn test_grad_0_1() {
        let f = |x: Dual<f64>| x*x + x.sin();
        assert!(f(Dual{ real: 0.0, grad: 1.0 }).grad < 1.1);
        assert!(f(Dual{ real: 0.0, grad: 1.0 }).grad > 0.9);
    }
    #[test]
    fn test_grad_pi_1() {
        let f = |x: Dual<f64>| x*x + x.sin();
        assert!(f(Dual{ real: 3.14, grad: 1.0 }).grad < 5.3);
        assert!(f(Dual{ real: 3.14, grad: 1.0 }).grad > 5.2);
    }
}
