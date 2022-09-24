def gradientdecent():
	x = [(1, 0, 3), (1, 1, 3), (1, 2, 3), (1, 3, 2), (1, 4, 4)]
	y = [95, 97, 75, 60, 49]
	# 假设y与x呈现线性关系，即hθ(x)=θX，J(θ)=(Xθ−Y)T(Xθ−Y)
	epsilon = 0.00001  # 初始化算法阈值为0.001
	theta = [0, 0, 0]  # 初始化theta为0
	alpha = 0.01  # 初始化步长为0.01
	error0 = 0  # 残差基准值初始化为0
	# 参数更新迭代过程
	while True:
		for i in range(len(x)):  # 更新参数theta
			theta[0] -= alpha*x[i][0]*(theta[0]+theta[1]*x[i][1]+theta[2]*x[i][2]-y[i])
			theta[1] -= alpha*x[i][1]*(theta[0]+theta[1]*x[i][1]+theta[2]*x[i][2]-y[i])
			theta[2] -= alpha*x[i][2]*(theta[0]+theta[1]*x[i][1]+theta[2]*x[i][2]-y[i])
		error1 = 0
		for j in range(len(x)):  # 带入真实数据，计算残差
			error1 += (y[j]-(theta[0]+theta[1]*x[j][1]+theta[2]*x[j][2]))**2/2
		if abs(error1-error0) < epsilon:  # 判断残差是否小于阈值
			break  # 收敛，结束
		else:
			error0 = error1  # 不收敛，还原残差，继续迭代
	print(theta[0], theta[1], theta[2])


gradientdecent()
