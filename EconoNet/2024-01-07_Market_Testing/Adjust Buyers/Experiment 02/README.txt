T = 100
B = 10 # bootstrap number of market simulations per iteration

price_adjust_method = 'step_random'

initialize_money = 'set'
M = 20

NA = 1

n = 2

QA1 = 10
QA2 = 0

QB1 = 0
QB2 = 0

initialize_prices = 'random'


QA_0 = np.array([QA1, QA2])
QB_0 = np.array([QB1, QB2])

QA = np.copy(QA_0)
QB = np.copy(QB_0)

qA = np.array([0, 0])
qB = np.array([0, 0])

cA = np.array([0, 0])
cB = np.array([1, 0])

DA = cA.copy()
DB = cB.copy()

dN = 1
NB_list = np.arange(1, 50+dN, dN)

NB_list = np.array(NB_list)