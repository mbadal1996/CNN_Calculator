# CNN_Calculator
CNN_Calculator aids in correctly constructing a CNN architecture for a given input image size.

======================================================================
CNN Calculator v1.0

Comments:

The purpose of this small code is to allow the user to determine some
parameters required in a CNN for a given input image size. These
parameters change depending on the input image size and the architecture
chosen. This sample CNN can be changed/enlarged but the basic idea remains.

In particular it computes the input dimension (A*B*C) to the first fully
connected (linear) layer near the end of the CNN. This can be tricky
and time consuming to do by hand since filter size, stride length, etc.
all affect this input dimension. But this calculator simplifies the process
by using the built-in operations in Pytorch.

After the example input image is fed into the CNN, the code outputs A,B,C 
which form the input dimension (of the first linear layer) as product: A*B*C

NOTE: The input image can have square or rectangular dimension.

The user should look for A*B*C in the code for further any clarification.


 ======================================================================
