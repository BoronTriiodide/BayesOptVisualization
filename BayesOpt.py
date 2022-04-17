import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import norm
from scipy.optimize import minimize
import timeit
import math
import sympy
from matplotlib import pyplot as plt

class BayesianOptimizer:
	DEFAULT_RANDOM_SAMPLES = 20
	DEFAULT_BAYESIAN_SAMPLES = 200
	GRAD_OPT_RETRIES = 1
	GRAD_OPT_MAX_EVAL = 100
	COVARIANCE_SCALE = 0.005
	COV_DIAGONAL_MOD = 1e-7
	OUTSIDE_DOMAIN_REWARD = -1000.0

	param_domain = []
	problem = None
	oldSamplePoints = np.array([[]])
	oldSampleResults = np.array([])
	oldSampleTasks = np.array([])
	optimum = ([], OUTSIDE_DOMAIN_REWARD)

	relatedTasks = 0
	task_covariance = {}

	problemEvalRuntime = 0
	gradOptRuntime = 0

	def __init__(self, param_domain, problem, seed = 0, visualizer = None):
		self.param_domain = param_domain
		self.problem = problem
		self.oldSamplePoints = np.empty((0,len(param_domain)))
		self.oldSampleResults = np.empty((0))
		self.oldSampleTasks = np.empty((0))
		task_self_covariance = {0: 1.0}
		self.task_covariance[0] = task_self_covariance
		self.visualizer = visualizer
		self.prev_optima = []
		np.random.seed(seed)
		np.set_printoptions(threshold=np.inf)

	def assignVariance(self, var):
		self.task_covariance[0][0] = var

	def getRandomSampleIndex(self):
		return np.random.randint(0, high=len(self.oldSamplePoints))

	def getRandomSamplePointInDomain(self):
		point = []
		for constraint in self.param_domain:
			point.append(np.random.uniform(low=constraint[0], high=constraint[1]))
		return np.array(point)

	def addRelatedTask(self, samplePoints, sampleResults, taskCovariance, taskVariance):
		self.relatedTasks += 1
		taskRelations = {}
		for i in range(self.relatedTasks):
			self.task_covariance[i][self.relatedTasks] = self.task_covariance[0][i]*taskCovariance/self.task_covariance[0][0]
			taskRelations[i] = self.task_covariance[0][i]*taskCovariance/self.task_covariance[0][0]
		taskRelations[self.relatedTasks] = taskVariance
		self.task_covariance[self.relatedTasks] = taskRelations

		self.oldSamplePoints = np.vstack((self.oldSamplePoints, samplePoints))
		self.oldSampleTasks = np.append(self.oldSampleTasks, np.array([self.relatedTasks]*sampleResults.shape[0]), axis=0)
		self.oldSampleResults = np.append(self.oldSampleResults, sampleResults, axis=0)

	def getCovariance(self, point1, task1, point2, task2):
		return self.task_covariance[task1][task2]*np.exp(-1*(np.linalg.norm(point1-point2)**2)/self.COVARIANCE_SCALE)

	def getTaskKernel(self):
		taskKernel = np.zeros((len(self.oldSampleTasks), len(self.oldSampleTasks)))
		for i in range(len(self.oldSampleTasks)):
			for j in range(len(self.oldSampleTasks)):
				taskKernel[i,j] = self.task_covariance[self.oldSampleTasks[i]][self.oldSampleTasks[j]]
		#if self.relatedTasks > 0:
		#	print(taskKernel)
		return taskKernel

	def getTaskToCurrentKernel(self):
		taskToCurrent = np.zeros((len(self.oldSampleTasks)))
		for i in range(len(self.oldSampleTasks)):
			taskToCurrent[i] = self.task_covariance[0][self.oldSampleTasks[i]]
		return taskToCurrent

	def getCovarianceKernel(self, taskKernel):
		a = self.oldSamplePoints
		b = a.reshape(a.shape[0], 1, a.shape[1])
		kernel = np.exp(-1*np.einsum('ijk, ijk -> ij', a-b, a-b)/self.COVARIANCE_SCALE)
		#if self.relatedTasks > 0:
		#	print(kernel)
		return taskKernel * kernel + np.eye(kernel.shape[0])*self.COV_DIAGONAL_MOD

	def getLinearlyDependantRows(self, kernel):
		_, inds = sympy.Matrix(kernel).T.rref()
		deps = []
		for i in range(kernel.shape[0]):
			if i not in inds:
				deps.append(i)
		return deps

	def getCovarianceKernelCholesky(self, covKernel):
		#if self.relatedTasks > 0:
		#	print(covKernel)
		try:
			cho_fact = cho_factor(covKernel)
			return cho_fact, True
		except np.linalg.LinAlgError as e:
			deps = self.getLinearlyDependantRows(covKernel)
			print("Dependent rows: ", deps)
			print(covKernel[deps])
			leading_minor = int(str(e).split('-')[0])-1
			print(self.oldSamplePoints.shape)
			self.oldSamplePoints = np.delete(self.oldSamplePoints, (leading_minor), axis=0)
			print(self.oldSamplePoints.shape)
			print(self.oldSampleResults.shape)
			self.oldSampleResults = np.delete(self.oldSampleResults, (leading_minor), axis=0)
			print(self.oldSampleResults.shape)
			self.oldSampleTasks = np.delete(self.oldSampleTasks, (leading_minor), axis=0)
			return None, False

	def getRelCovKernel(self, newPoint, taskToCurrentKernel):
		a = newPoint
		b = self.oldSamplePoints
		return taskToCurrentKernel * np.exp(-1*np.einsum('ij, ij -> i', a-b, a-b)/self.COVARIANCE_SCALE)

	def getRelCovDeriv(self, newPoint, var, relCovKernel):
		a = self.oldSamplePoints
		return -2.0 * (newPoint[var] / self.COVARIANCE_SCALE - a[:,var]) * relCovKernel

	def getPriorMean(self, relCovariance, leftCovSolution):
		return np.dot(leftCovSolution, relCovariance)

	def getPriorVariance(self, newPoint, relCov, covCholesky, varScale):
		inherentVar = self.getCovariance(newPoint, 0, newPoint, 0)
		varReduction = np.dot(cho_solve(covCholesky, relCov), relCov)
		return varScale * (inherentVar - varReduction)

	def getVarScaleMLE(self, leftCovSolution):
		#resultsVec = self.oldSampleResults
		#return np.dot(leftCovSolution, resultsVec) / len(self.oldSampleResults)
		return 1

	def getStandardNormalDensity(self, arg):
		return norm(0, 1).pdf(arg)

	def getStandardCumDist(self, arg):
		return norm(0, 1).cdf(arg)

	def getMeanAndVarFs(self):
		cholesky_success = False
		covCholesky = None
		while not cholesky_success:
			fullTaskKernel = self.getTaskKernel()
			selfCov = self.getCovarianceKernel(fullTaskKernel)
			covCholesky, cholesky_success = self.getCovarianceKernelCholesky(selfCov)
		taskToCurrentKernel = self.getTaskToCurrentKernel()
		leftCovSolution = cho_solve(covCholesky, self.oldSampleResults)
		varScale = self.getVarScaleMLE(leftCovSolution)
		def priorMean(samplePoint):
			relCov = self.getRelCovKernel(samplePoint, taskToCurrentKernel)
			return self.getPriorMean(relCov, leftCovSolution)
		def priorVar(samplePoint):
			relCov = self.getRelCovKernel(samplePoint, taskToCurrentKernel)
			return self.getPriorVariance(samplePoint, relCov, covCholesky, varScale)
		return priorMean, priorVar

	def getNegExpectedImprovement(self, samplePoint, leftCovSolution, covCholesky, varScale, taskToCurrentKernel):
		relCov = self.getRelCovKernel(samplePoint, taskToCurrentKernel)
		priorMean = self.getPriorMean(relCov, leftCovSolution)
		priorStdDev = self.getPriorVariance(samplePoint, relCov, covCholesky, varScale)**0.5
		normalArg = (priorMean - self.optimum[1])/(max(1e-7, priorStdDev))
		return -1*(normalArg*priorStdDev + priorStdDev*self.getStandardNormalDensity(normalArg) - normalArg*priorStdDev*self.getStandardCumDist(normalArg))

	def getNegEIJacobian(self, samplePoint, leftCovSolution, covCholesky, varScale, taskToCurrentKernel):
		relCov = self.getRelCovKernel(samplePoint, taskToCurrentKernel)
		priorMean = self.getPriorMean(relCov, leftCovSolution)
		priorStdDev = self.getPriorVariance(samplePoint, relCov, covCholesky, varScale)**0.5
		normalArg = (priorMean - self.optimum[1])/(max(1e-7, priorStdDev))
		jacobian = np.zeros((len(samplePoint)))
		for i in range(len(samplePoint)):
			relCovDeriv = self.getRelCovDeriv(samplePoint, i, relCov)
			priorMeanDeriv = self.getPriorMean(relCovDeriv, leftCovSolution)
			priorDevDeriv = self.getPriorVariance(samplePoint, relCovDeriv, covCholesky, varScale) / (2 * priorStdDev)
			jacobian[i] = -1*(priorMeanDeriv + (priorDevDeriv * self.getStandardNormalDensity(normalArg) + (normalArg**2 - normalArg*priorMeanDeriv)*self.getStandardNormalDensity(normalArg)) - (priorMeanDeriv * self.getStandardCumDist(normalArg) + self.getStandardNormalDensity(normalArg)*(normalArg*priorMeanDeriv - normalArg**2)))
		return jacobian

	def getEIAcquisition(self):
		cholesky_success = False
		covCholesky = None
		while not cholesky_success:
			fullTaskKernel = self.getTaskKernel()
			selfCov = self.getCovarianceKernel(fullTaskKernel)
			covCholesky, cholesky_success = self.getCovarianceKernelCholesky(selfCov)
		taskToCurrentKernel = self.getTaskToCurrentKernel()
		leftCovSolution = cho_solve(covCholesky, self.oldSampleResults)
		varScale = self.getVarScaleMLE(leftCovSolution)
		def negEI(samplePoint):
			return self.getNegExpectedImprovement(samplePoint, leftCovSolution, covCholesky, varScale, taskToCurrentKernel)
		def negEIJacobian(samplePoint):
			return self.getNegEIJacobian(samplePoint, leftCovSolution, covCholesky, varScale, taskToCurrentKernel)
		return (negEI, negEIJacobian)

	def nextSamplePoint(self):
		gradOptStart = timeit.default_timer()
		negEI, negEIJacobian = self.getEIAcquisition()
		next_point = []
		for i in range(self.GRAD_OPT_RETRIES):
			startingPoint = self.getRandomSamplePointInDomain()
			proposed = minimize(negEI, startingPoint, method='BFGS', jac=negEIJacobian, options={'maxiter':self.GRAD_OPT_MAX_EVAL})
			next_point = proposed.x
			if negEI(proposed.x) < 0:
				break
		self.gradOptRuntime += timeit.default_timer() - gradOptStart
		return next_point

	def nextSamplePointVisualized(self):
		ns = np.arange(self.param_domain[0][0], self.param_domain[0][1], 0.001)
		fs = []
		for n in ns:
			fs.append(self.problem([n]))
		self.visualizer[0].plot(ns, fs, color="blue")
		self.visualizer[0].scatter(self.oldSamplePoints, self.oldSampleResults, color="black")
		self.visualizer[0].title.set_text("f(x) = x^2 sin^6 (5pi x)")

		mean, var = self.getMeanAndVarFs()
		means, variances = [], []
		for n in ns:
			means.append(mean(np.array([n])))
			variances.append(var(np.array([n])))
		self.visualizer[1].plot(ns, means, color="blue")
		self.visualizer[1].fill_between(ns, [means[n] - 2*(variances[n])**0.5 for n in range(len(means))], [means[n] + 2*(variances[n])**0.5 for n in range(len(means))], color="blue", alpha=0.2)
		self.visualizer[1].title.set_text("Gaussian Process Mean and 95% CI")

		negEI, negEIJac = self.getEIAcquisition()
		EI_display = []
		opt = (0, -1*np.inf)
		for n in ns:
			EI_display.append(-1*negEI(np.array([n])))
			if -1*negEI(np.array([n])) > opt[1]:
				opt = (np.array([n]), -1*negEI(np.array([n])))
		self.visualizer[2].plot(ns, EI_display, color="blue")
		self.visualizer[2].scatter([opt[0]], [opt[1]], color="black")
		self.visualizer[2].title.set_text("Expected Improvement and The Next Point to Sample")

		self.visualizer[3].plot([n for n in range(len(self.prev_optima))], self.prev_optima, color="black")
		self.visualizer[3].title.set_text("The Optimum Found by Time Step")

		return opt[0]

	def check_domain(self, point):
		for coord, constraint in zip(point, self.param_domain):
			if coord < constraint[0] or coord > constraint[1]:
				return False
		return True

	def applySample(self, point):
		sampleResult = 0
		if self.check_domain(point):
			evalStart = timeit.default_timer()
			sampleResult = self.problem(point)
			self.problemEvalRuntime += timeit.default_timer() - evalStart
		else:
			sampleResult = self.OUTSIDE_DOMAIN_REWARD
		self.oldSamplePoints = np.vstack((self.oldSamplePoints, np.array(point)))
		self.oldSampleTasks = np.append(self.oldSampleTasks, np.array([0]), axis=0)
		self.oldSampleResults = np.append(self.oldSampleResults, np.array([sampleResult]), axis=0)
		if (sampleResult > self.optimum[1]):
			self.optimum = (point, sampleResult)
		self.prev_optima.append(self.optimum[1])

	def addPointAndResult(self, point, result):
		self.oldSamplePoints = np.vstack((self.oldSamplePoints, np.array(point)))
		self.oldSampleTasks = np.append(self.oldSampleTasks, np.array([0]), axis=0)
		self.oldSampleResults = np.append(self.oldSampleResults, np.array([result]), axis=0)
		if (result > self.optimum[1]):
			self.optimum = (point, result)

	def optimize(self):
		for i in range(self.DEFAULT_RANDOM_SAMPLES):
			point = self.getRandomSamplePointInDomain()
			self.applySample(point)
		for i in range(self.DEFAULT_BAYESIAN_SAMPLES):
			point = self.nextSamplePoint()
			self.applySample(point)
		return self.optimum

	def optimize_direct(self, steps):
		for i in range(steps):
			if self.visualizer is None:
				point = self.nextSamplePoint()
			elif self.oldSampleResults.shape[0] == 0:
				point = self.getRandomSamplePointInDomain()
			else:
				point = self.nextSamplePointVisualized()
			self.applySample(point)
		return self.optimum

def doVisualization(seed):
	def prob(x):
		return x[0]**2 * math.sin(5*math.pi *x[0])**6
	param_dom = [(0,1.0)]
	fig, ax = plt.subplots(4)
	plt.ion()
	plt.tight_layout()
	plt.show()
	opt = BayesianOptimizer(param_dom, prob, 0, ax)
	for i in range(20):
		_ = input()
		for axis in ax:
			axis.clear()
		opt.optimize_direct(1)
		plt.draw()
		plt.pause(0.01)

def one_dim_test():
	def prob(x):
		return x[0]**2 * math.sin(5*math.pi *x[0])**6
	param_dom = [(0,1.0)]
	opt = BayesianOptimizer(param_dom, prob)
	start = timeit.default_timer()
	optimum = opt.optimize()
	print("Optimum found: ", optimum)
	print("Runtime: ", timeit.default_timer() - start)

def two_dim_test():
	def prob(x):
		return -1*(x[0] - 1.5)**2 - (x[1] + 3)**2 - (x[2])**2 - (x[3] - 4)**2
	param_dom = [(-5,5),(-5,5),(-5,5),(-5,5),(-5,5)]
	opt = BayesianOptimizer(param_dom, prob)
	start = timeit.default_timer()
	optimum = opt.optimize()
	print("Optimum found: ", optimum)
	print("Runtime: ", timeit.default_timer() - start)