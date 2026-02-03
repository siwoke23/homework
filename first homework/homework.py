# 第一周作业
"""
第一部分：
ReLU变体：
1.Leaky ReLU:if x >=0  f(x)=x,otherwise  f(x)=alpha*x,0<=alpha<=1;
2.ELU函数，if x > 0  f(x)=x,otherwise  f(x)=alpha*(exp(x)-1);
3.SELU函数，if x > 0  f(x)=lambda*x,otherwise  f(x)=lambda*(alpha(exp(x)-1))
4.GELU函数，f(x)=x/2(1+Tanh((2/PI)**(-1/2)*(x+0.044715*x**3)))

激活函数还有:
1.sigmoid函数，f(x)=1/(1+exp(-x))；
2.阶跃函数，以0为界，输出从0切换为1(或者从1切换为0);
3.线性函数，f(x)=x；
4.Softplus函数，f(x)=ln(1+exp(x));
5.Swish函数，f(x)=x*sigma(beta*x)=x/(1+exp(-beta*x))

  逆向编码（Backward Coding）是通用编程的开发思维模式，核心是输出倒推实现步骤
  是写代码的程序步骤，而不是深度学习的反向传播，二者都使用了逆向思维，但本质是不同的
"""