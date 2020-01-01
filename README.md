# ad

Automatic differentiation in Rust.

## Usage

 ```rust
use ad::forward::Dual;
 
fn main(){
    let f = |x: Dual<f64>, y: Dual<f64>| Dual::c(2.0) * x * x + Dual::c(3.0) * x * y + Dual::c(4.0);
    println!("{:?}", f(Dual{
        real: 1.0,
        grad: 1.0,
    }, Dual{
        real: 2.0,
        grad: 0.0,
    })); // Dual { real: 12.0, grad: 10.0 }
    assert!(false);
}
```


## License

[BSD-3-Clause](https://opensource.org/licenses/BSD-3-Clause)
