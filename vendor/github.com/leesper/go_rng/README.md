# go_rng

[![Build Status](https://travis-ci.org/leesper/go_rng.svg?branch=master)](https://travis-ci.org/leesper/go_rng)
[![CircleCI](https://circleci.com/gh/leesper/go_rng.svg?style=svg)](https://circleci.com/gh/leesper/go_rng)
[![GitHub stars](https://img.shields.io/github/stars/leesper/go_rng.svg)](https://github.com/leesper/go_rng/stargazers)
[![GitHub license](https://img.shields.io/badge/license-Apache%202-blue.svg)](https://raw.githubusercontent.com/leesper/go_rng/master/LICENSE)
[![GoDoc](https://godoc.org/github.com/leesper/go_rng?status.svg)](http://godoc.org/github.com/leesper/go_rng)

A pseudo-random number generator written in Golang v1.3 伪随机数生成器库的Go语言实现

## Features

### Inspired by:
* [StdRandom.java](http://introcs.cs.princeton.edu/java/stdlib/StdRandom.java.html)
* [Numerical Recipes](http://www.nr.com/)
* [Random number generation](http://en.wikipedia.org/wiki/Random_number_generation)
* [Quantile function](http://en.wikipedia.org/wiki/Quantile_function)
* [Monte Carlo method](http://en.wikipedia.org/wiki/Monte_Carlo_method)
* [Pseudo-random number sampling](http://en.wikipedia.org/wiki/Pseudo-random_number_sampling)
* [Inverse transform sampling](http://en.wikipedia.org/wiki/Inverse_transform_sampling)

### Supported Distributions and Functionalities:
均匀分布      Uniform Distribution <br />
伯努利分布    Bernoulli Distribution <br />
卡方分布      Chi-Squared Distribution <br />
Gamma分布     Gamma Distribution <br />
Beta分布      Beta Distribution <br />
费舍尔F分布   Fisher's F Distribution <br />
柯西分布      Cauchy Distribution <br />
韦伯分布      Weibull Distribution <br />
Pareto分布    Pareto Distribution <br />
对数高斯分布  Log Normal Distribution <br />
指数分布      Exponential Distribution <br />
学生T分布     Student's t-Distribution <br />
二项分布      Binomial Distribution <br />
泊松分布      Poisson Distribution <br />
几何分布      Geometric Distribution <br />
高斯分布      Gaussian Distribution <br />
逻辑分布      Logistic Distribution <br />
狄利克雷分布  Dirichlet Distribution <br />

## Requirements

* Golang 1.7 and above

## Installation

`go get -u -v github.com/leesper/go_rng`

## Usage

```go
func TestGaussianGenerator(t *testing.T) {
	fmt.Println("=====Testing for GaussianGenerator begin=====")
	grng := NewGaussianGenerator(time.Now().UnixNano())
	fmt.Println("Gaussian(5.0, 2.0): ")
	hist := map[int64]int{}
	for i := 0; i < 10000; i++ {
		hist[int64(grng.Gaussian(5.0, 2.0))]++
	}

	keys := []int64{}
	for k := range hist {
		keys = append(keys, k)
	}
	SortInt64Slice(keys)

	for _, key := range keys {
		fmt.Printf("%d:\t%s\n", key, strings.Repeat("*", hist[key]/200))
	}

	fmt.Println("=====Testing for GaussianGenerator end=====")
	fmt.Println()
}
```
output:
```
=====Testing for GaussianGenerator begin=====
Gaussian(5.0, 2.0):
-2:
-1:
0:	*
1:	**
2:	****
3:	*******
4:	*********
5:	*********
6:	*******
7:	****
8:	**
9:
10:
11:
12:
=====Testing for GaussianGenerator end=====
```

## Authors and acknowledgment

* [Danny Patrie](https://github.com/dpatrie)
* [Akihiro Suda](https://github.com/AkihiroSuda)
* [Sho IIZUKA](https://github.com/arosh)
* [Paul Bohm](https://github.com/enki)

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. Please make sure to update tests as appropriate.
