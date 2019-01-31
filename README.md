
# Solving Systems of Linear Equations with NumPy - Code Along

## Introduction

In this lesson, we shall see how to solve a system of linear equations using matrix algebra and numpy.  We shall learn about the identity matrix and inverse matrices, which have some unique properties which can be used to solve for unknown values in systems of linear equations. We will discover how to create an identity matrix and also how to calculate the inverse of a matrix in Python and Numpy. 

## Objectives

You will be able to:

* Understand and describe identity matrix and its role in linear equations
* Calculate Inverse of a matrix in order to solve linear problems
* Use the Matrix algebra and Numpy skills to solve a system of linear equations

## Identity Matrix

An identity matrix is a matrix whose dot product with another matrix M equals the same matrix M.

The identity matrix is a square matrix which contains **1s** along the major diagonal (from the top left to the bottom right), while all its other entries are **0s**. Following images helps identify "Major" diagonal of a given matrix. 
![](https://www.tutorialride.com/images/c-array-programs/major-diagonal.jpeg)

An identity matrix for above matrix, is thus containing all 1s along this diagnoal and 0s everywhere else as shown below:

```
1 0 0 
0 1 0 
0 0 1
```


This would be called a 3x3 Identity matrix. The `n×n` Identity matrix is ususally denoted by **I<sub>n</sub>** which is a matrix with n rows and n columns. Other examples include 2x2 , 4x4 Identity matrices etc. 


Identity Matrix is also called Unit Matrix or Elementary Matrix

### Dot-Product of Given Matrix and its Identity Matrix

Let's try to multiply a matrix with its identity matrix and check the output. Let's start with the co-efficient matrix from our previous problem:
```
2 1 
3 4
```


Our identity matrix for this matrix would look like:
```
1 0 
0 1
```

Let's take the dot-product for these two matrices as shown below:
```python
import numpy as np
A = np.array([[2,1],[3,4]])
I = np.array([[1,0],[0,1]])
print(I.dot(A))
print('\n', A.dot(I))
```


```python
# Code here 
```

So we see that the dot-product of any matrix and the appropriate identity matrix is always the original matrix, regardless of the order in which the multiplication was performed! In other words, 
> **`A . I = I . A = A`**

NumPy comes with a built-in function `np.identity()` for producing an identity matrix. Just pass it the dimension (numnber of rows or columns) as the argument. Optionally tell it to output elements as integers to keep the output as integers (It'll create a float based Identity matrix otherwise):
```python
print (np.identity(4, dtype=int))
print (np.identity(5, dtype=int))
```


```python
# Code here 
```

## Inverse Matrix

The *Inverse* of a square matrix *A*, sometimes called a *reciprocal matrix*, is a matrix *A<sup>-1</sup>* such that

> **A . A<sup>-1</sup> = I**	

where **I** is the Identity matrix. 


The inverse of a matrix is analogous to taking raciprocal of a number and multiplying by itself to get a 1, e.g. 5 * 5<sup>-1</sup> = 1. Let's see how to get inverse of a matrix in numpy. `numpy.linalg.inv(a)` takes in a matrix a and calculates its inverse as shown below.

```python
A = np.array([[4,2,1],[4,8,3],[1,1,0]])
A_inv = np.linalg.inv(A)
print(A_inv)
```


```python
# Code here 
```

This is great. So according to the principle shown above, if we multiply A with A<sup>-1</sup>, we should get an identity matrix I in the output. 

```python
A_product = np.dot(A,A_inv)
A_product
```


```python
# Code here 
```

Note that this was meant to return the identity matrix. We have all 1s along major diagonal but that the float operations returned not zeros but numbers very close to zero at other places. It is also trivial to think that a matrix of all zeros has no inverse (we need some 1s in the output). Numpy has a `np.matrix.round` function to convert each element of above matrix into a decimal form. 

```python
np.matrix.round(A_product)
```


```python
# Code here 
```

So this looks more like an identity matrix that we saw earlier. The negative signs remain after rounding off as the original small values were negative. This, however, wont effect computation in any way. 

## Why Do We Need an Inverse?

Because with matrices we can not divide! **There is no concept of dividing by a matrix**. But we can multiply by an inverse, which achieves the same thing.

Imagine we have a problem:  

> "How do I share 10 apples with 2 people?"

We can divide 10 by 2 - OR - We can take the reciprocal of 2 (which is 0.5), so we answer:

10 × 0.5 = 5 means They get 5 apples each.

We use the very same idea here and this can be used to solve a system of linear equation in the problems we saw earlier in the section where: 

> **A . X = B** (remember `A` is the matrix of co-efficients, `X` is the unknown variable and `B` is the output)

Say we want to find matrix X, when we already know matrix A and B:

It would've been great if we could divide both sides by A to get `X = B / A`, but remember we can't divide. We can achieve this if we multiply both sides by A<sup>-1</sup>, as shown below:

> **X . A . A<sup>-1</sup> = B . A<sup>-1</sup>**

From above , we that A . A<sup>-1</sup> = I, so:

> **X . I = B . A<sup>-1</sup>**

We can remove I (because multiplying with identity matrix doesn't change a matrix). so:

> **X = B A<sup>-1</sup> **

And there we have it, our answer. 

## Solve a System of Equations with Matrix Algebra. 

Now that we know everything about converting a simple real world problem into matrix format, and steps to solve the problem, let's try it out with our apples and bananas problem from very first lesson. 
![](ab.png)

So first we'll need to calculate the inverse of the square matrix containing coefficient values.
```python
# Define A and B 
A = np.matrix([[2, 1], [3, 4]])
B = np.matrix([35,65])

# Take the inverse of Matrix A 
A_inv = np.linalg.inv(A)
A_inv
```


```python
# Code here 
```

We can now take a dot product of `A_inv` and `B`. Also, as we want the output in the vector format (containing one column and two rows), we would need to transpose the matrix `B` to satisfy the multiplication rule we saw in the previous lesson.

> **The product of an M x N matrix and an N x K matrix is an M x K matrix. The new matrix takes the rows of the 1st and columns of the 2nd**

```python
# Check the shape of B before after transposing
print(B.shape)
B = B.T
print (B.shape)
B
```


```python
# Code here 
```

Now we can easily calculate X as below:

```python
X = A_inv.dot(B)
X
```


```python
# Code here 
```

So there we have it, our answer. We can see that the prices of apples and bananas have been calculated as 15p / apple and 5p / banana, and these values satisfy both equations. 

The dot product of A and X should give us the matrix B. Let's try it:
```python
print(A.dot(X))
print (B)
```


```python
# Code here 
```

Success . 

#### `numpy.linalg.solve()` to solve a system of linear equations

Numpy has a built in function to solve such equations as `numpy.linalg.solve(a,b)` which takes in matrices in the correct orientation, and gives the answer by calculating the inverse. Here is how we use it. 

```python
# Use Numpy's built in function solve() to solve linear equations
x = np.linalg.solve(A,B)
x
```


```python
# Code here 
```

## Further Reading

* [Youtube: Solving System of Linear Equations using Python](https://youtu.be/AqIrdW2-K6k)
* [Inverse of a Matrix](http://www.mathwords.com/i/inverse_of_a_matrix.htm)
* [Don't invert that Matrix](https://www.johndcook.com/blog/2010/01/19/dont-invert-that-matrix/)

## Summary

In this lesson, we saw how to calculate inverse of a matrix in order to solve a system of linear equations. We applied the skills learnt on the simple problem that we defined earlier. The result of our calculations helped us get unit values of variables that satisfy both equations. In the next lab, we shall go through some other similar problems. 
