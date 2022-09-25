import pandas as pd
import numpy as np

import sklearn
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score
import optunity
import optunity.metrics

import data_loader
import utils
import preprocessing


def _preprocess_svm2(train: pd.DataFrame, test: pd.DataFrame, validation: pd.DataFrame):
	window_length = 5
	dfs = [train, test, validation]
	xs = []
	ys = []

	for df in dfs:
		x = []
		y = []
		window_offset_count = 0  # skip the first window_length rows
		i = 0
		indexes = []
		for index in df.index:
			window_offset_count += 1
			indexes.append(index)
			if window_offset_count > window_length:
				window = []
				for j in range(0, window_length):
					window.append(df.loc[indexes[i - j], 'WTI'])
				x.append(window)
				y.append(df.loc[index, 'y'])
			i += 1
		x_np = np.array(x)
		xs.append(x_np)
		ys.append(y)

	return xs, ys


def train_model(x_train, y_train, kernel, C, logGamma, degree, coef0):
	"""A generic SVM training function, with arguments based on the chosen kernel."""
	if kernel == 'linear':
		model = sklearn.svm.SVC(kernel=kernel, C=C)
	elif kernel == 'poly':
		model = sklearn.svm.SVC(kernel=kernel, C=C, degree=degree, coef0=coef0)
	elif kernel == 'rbf':
		model = sklearn.svm.SVC(kernel=kernel, C=C, gamma=10 ** logGamma)
	else:
		print("Unknown kernel function: %s" % kernel)
		return
	model.fit(x_train, y_train)
	return model


def svm_tuned_auroc(x_train, y_train, x_test, y_test, kernel='linear', C=0, logGamma=0, degree=0, coef0=0):
	model = train_model(x_train, y_train, kernel, C, logGamma, degree, coef0)
	decision_values = model.decision_function(x_test)
	return optunity.metrics.roc_auc(y_test, decision_values)


def validate_svm1(x_val, y_val, model):
	predicted_val_y = []
	print('Testing...')
	for i in range(0, len(x_val)):
		predicted_val_y.append(model.predict([x_val[i]]))

	print('\tConfusion matrix: ')
	print(confusion_matrix(y_val, predicted_val_y))

	print('\tAccuracy:')
	print(accuracy_score(y_val, predicted_val_y))


df = data_loader.load_dataframe()
validation = data_loader.load_validation_set()

df_original = df.copy()
validation_original = validation.copy()

train, test, validation = preprocessing.prepare_for_classification(df, validation)

xs, ys = _preprocess_svm2(train, test, validation)
x_train = xs[0]
y_train = ys[0]
x_val = xs[2]
y_val = ys[2]

space = {'kernel': {'linear': {'C': [0, 2]},
                    'rbf': {'logGamma': [-5, 0], 'C': [0, 10]},
                    'poly': {'degree': [2, 5], 'C': [0, 5], 'coef0': [0, 2]}
                    }
         }

cv_decorator = optunity.cross_validated(x=x_train, y=y_train, num_folds=5)
svm_tuned_auroc = cv_decorator(svm_tuned_auroc)

optimal_svm_pars = {}
info = None
if not optimal_svm_pars:
	print('Tuning hyperparameters...')
	optimal_svm_pars, info, _ = optunity.maximize_structured(svm_tuned_auroc, space, num_evals=150)
	print("Optimal parameters" + str(optimal_svm_pars))
	print('INFO: ', info)
	print("AUROC of tuned SVM: %1.3f" % info.optimum)

# preparing hyperparameters
hyperparameters = {}
for item in optimal_svm_pars.items():
	if item[1] is not None:
		hyperparameters[item[0]] = item[1]

print('Training model...')
optimal_model = sklearn.svm.SVC(**hyperparameters).fit(x_train, y_train)
print('Validating model')
validate_svm1(x_val, y_val, optimal_model)
if info:
	df = optunity.call_log2dataframe(info.call_log)
	df.sort_values('value', ascending=False)

utils.ring_bell()

'''
Tuning hyperparameters...
Optimal parameters{'kernel': 'linear', 'C': 0.9215500860525165, 'coef0': None, 'degree': None, 'logGamma': None}
INFO:  NT(optimum=0.5277594406241983, stats={'num_evals': 150, 'time': 48454.14178579999}, call_log={'args': {'logGamma': [None, None, -1.73095703125, None, None, None, None, -3.13720703125, None, None, None, None, -1.9642903645833334, None, None, None, None, -3.3705403645833334, None, None, None, None, -2.197623697916667, None, None, None, None, -3.603873697916667, None, None, None, None, -1.9642903645833334, None, None, None, None, -3.3705403645833334, None, None, None, None, None, None, None, None, None, -3.13720703125, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], 'degree': [4.61181640625, None, None, 4.23681640625, None, None, 4.33056640625, None, 2.08056640625, 3.95556640625, 4.47181640625, None, None, 4.09681640625, None, None, 4.19056640625, None, None, 3.81556640625, 4.331816406250001, None, None, 3.95681640625, None, None, 4.050566406250001, None, None, 3.6755664062499998, None, None, None, 3.81681640625, None, None, 3.9105664062500005, None, None, 3.5355664062499996, None, None, 4.426816406250001, 3.67681640625, None, None, 3.7705664062500004, None, None, None, None, None, 4.286816406250002, 3.53681640625, None, None, 3.6305664062500003, 3.099508593326081, None, None, None, None, 4.146816406250002, None, None, None, None, 2.9595085933260807, None, None, None, None, 4.006816406250002, None, None, None, None, 2.8195085933260806, None, None, None, None, 3.866816406250002, None, None, None, None, 2.701067816266381, None, None, None, None, 3.726816406250002, None, None, None, None, 2.8102343811344492, None, None, None, None, 3.586816406250002, None, None, None, None, 2.9502343811344494, None, None, None, None, None, None, None, None, None, 3.040805207765059, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], 'coef0': [0.0029296875, None, None, 1.2529296875, None, None, 1.1904296875, None, 1.6904296875, 0.4404296875, 0.09626302083333332, None, None, 1.3462630208333333, None, None, 1.2837630208333333, None, None, 0.5337630208333333, 0.18959635416666665, None, None, 1.4395963541666665, None, None, 1.3770963541666665, None, None, 0.6270963541666665, None, None, None, 1.5329296874999998, None, None, 1.4704296874999998, None, None, 0.7204296874999998, None, None, 0.9395963541666665, 1.4395963541666665, None, None, 1.457127876743508, None, None, None, None, None, 0.8462630208333333, 1.3462630208333333, None, None, 1.3637945434101748, 0.7837630208333333, None, None, None, None, 0.7529296875, None, None, None, None, 0.6904296875, None, None, None, None, 0.6595963541666667, None, None, None, None, 0.5970963541666667, None, None, None, None, 0.5662630208333335, None, None, None, None, 0.5037630208333335, None, None, None, None, 0.47292968750000014, None, None, None, None, 0.45656775227656216, None, None, None, None, 0.4366732627420316, None, None, None, None, 0.45114927493333257, None, None, None, None, None, None, None, None, None, 0.4801792991671813, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], 'kernel': ['poly', 'linear', 'rbf', 'poly', 'linear', 'linear', 'poly', 'rbf', 'poly', 'poly', 'poly', 'linear', 'rbf', 'poly', 'linear', 'linear', 'poly', 'rbf', 'linear', 'poly', 'poly', 'linear', 'rbf', 'poly', 'linear', 'linear', 'poly', 'rbf', 'linear', 'poly', 'linear', 'linear', 'rbf', 'poly', 'linear', 'linear', 'poly', 'rbf', 'linear', 'poly', 'linear', 'linear', 'poly', 'poly', 'linear', 'linear', 'poly', 'rbf', 'linear', 'linear', 'linear', 'linear', 'poly', 'poly', 'linear', 'linear', 'poly', 'poly', 'linear', 'linear', 'linear', 'linear', 'poly', 'linear', 'linear', 'linear', 'linear', 'poly', 'linear', 'linear', 'linear', 'linear', 'poly', 'linear', 'linear', 'linear', 'linear', 'poly', 'linear', 'linear', 'linear', 'linear', 'poly', 'linear', 'linear', 'linear', 'linear', 'poly', 'linear', 'linear', 'linear', 'linear', 'poly', 'linear', 'linear', 'linear', 'linear', 'poly', 'linear', 'linear', 'linear', 'linear', 'poly', 'linear', 'linear', 'linear', 'linear', 'poly', 'linear', 'linear', 'linear', 'linear', 'linear', 'linear', 'linear', 'linear', 'linear', 'poly', 'linear', 'linear', 'linear', 'linear', 'linear', 'linear', 'linear', 'linear', 'linear', 'linear', 'linear', 'linear', 'linear', 'linear', 'linear', 'linear', 'linear', 'linear', 'linear', 'linear', 'linear', 'linear', 'linear', 'linear', 'linear', 'linear', 'linear', 'linear', 'linear', 'linear', 'linear', 'linear'], 'C': [1.70654296875, 0.5400390625, 9.5166015625, 2.33154296875, 1.0400390625, 0.1025390625, 0.61279296875, 7.3291015625, 4.36279296875, 1.23779296875, 1.9398763020833334, 0.4834848467294201, 9.049934895833333, 2.5648763020833334, 0.9467057291666667, 0.19587239583333332, 0.8461263020833333, 6.862434895833333, 1.5092057291666667, 1.4711263020833334, 2.173209635416667, 0.4269306309588402, 8.583268229166666, 2.798209635416667, 0.8533723958333335, 0.28920572916666665, 1.0794596354166666, 6.395768229166666, 1.4158723958333335, 1.7044596354166668, 1.5100390625000002, 0.5202639642921736, 8.1166015625, 3.03154296875, 0.9215500860525165, 0.38253906249999997, 1.31279296875, 5.9291015625, 1.3225390625000002, 1.9377929687500002, 1.416705729166667, 0.6135972976255069, 2.0148763020833336, 3.2648763020833336, 0.9897277762716996, 0.4758723958333333, 1.5461263020833333, 5.462434895833333, 1.229205729166667, 0.8525390625, 1.3233723958333337, 0.7069306309588401, 2.248209635416667, 3.498209635416667, 0.9952089709807983, 0.5692057291666666, 1.7794596354166667, 3.029459635416667, 1.1358723958333337, 0.9458723958333333, 1.2300390625000004, 0.8002639642921734, 2.4815429687500004, 0.6000390624999998, 0.901875637647465, 0.6625390624999998, 0.8380550296071301, 3.2627929687500004, 1.0425390625000004, 1.0077900854864732, 1.1367057291666671, 0.8935972976255067, 2.7148763020833337, 0.6933723958333331, 0.8414234967427432, 0.7558723958333331, 0.8366248919849203, 3.4961263020833337, 0.9492057291666671, 1.001860965125697, 1.0433723958333339, 0.9869306309588399, 2.948209635416667, 0.7867057291666664, 0.8795052577725627, 0.8492057291666664, 0.9299582253182537, 3.729459635416667, 0.8558723958333339, 0.9085276317923636, 0.9500390625000006, 0.9988299291894596, 3.1815429687500005, 0.8800390624999996, 0.9728385911058961, 0.9425390624999996, 1.0170217411380087, 3.9627929687500005, 0.7724028197913555, 0.8373059847673078, 0.8567057291666673, 0.9506283086335298, 3.414876302083334, 0.9733723958333329, 0.9846633424883556, 1.0230476319288861, 0.9938552286753884, 4.196126302083334, 0.7414060498612357, 0.7807493270094621, 0.8309767347867505, 0.8958145687305852, 0.8358902352262381, 0.9996045558398289, 0.8913300091550223, 1.036100864384951, 0.9005218953420551, 4.408811720758561, 0.8347393831945691, 0.7533369075998403, 0.9243100681200838, 0.874570292101256, 0.9292235685595713, 0.9314141669022743, 0.8451450040804829, 0.9427675310516177, 0.8081534057491895, 0.9015354662065157, 0.9280727165279024, 0.8466702409331737, 1.0156990345382824, 0.9068241392440403, 1.0197756885126104, 0.8536550651095128, 0.8428376707862194, 0.8494341977182844, 0.7635138862406716, 0.9569147301173593, 1.0181918392458074, 0.9400035742665069, 1.0014541059651552, 0.955246161699859, 1.0241629543578725, 0.8905467295489865, 0.9361710041195528, 0.850564957229419, 0.8568472195740049, 0.9881886551776649, 0.9258575124363753, 0.9980288710259491]}, 'values': [0.5037651088382513, 0.5248854738523145, 0.48576181084229, 0.4977010108317462, 0.5247174401916468, 0.5227576347815985, 0.49481665666024, 0.5004890915224006, 0.4986623190991903, 0.5001942760660557, 0.5004025229815668, 0.5261902919478506, 0.49308391700563536, 0.4928464676272453, 0.5250639576600061, 0.5252655809836632, 0.4912557817834344, 0.4955556352501702, 0.5255702232276379, 0.5046466091872209, 0.5005933788837749, 0.523759410612121, 0.4953260474919542, 0.5017275185408133, 0.5250737810006567, 0.5241731483021246, 0.4966482207323081, 0.5001008145062469, 0.5267830948694532, 0.49874056381730547, 0.5265655983075486, 0.5240770144894605, 0.4931460622734131, 0.5009057396370007, 0.5277594406241983, 0.5237756182736988, 0.4984328737346651, 0.5032030225121262, 0.5247926085126321, 0.5019671268786813, 0.523766612144436, 0.5236024783640716, 0.49694781520037645, 0.4990563431116598, 0.5259854236183215, 0.5249834690990274, 0.5017627759994472, 0.5006675883416156, 0.5258066073691439, 0.5255074714692489, 0.5228461654168967, 0.5259521873819433, 0.49809988189363663, 0.5058936582616036, 0.5251013127431882, 0.5273648380563387, 0.49968609936896574, 0.4950915067837256, 0.5264219092577971, 0.5258403854131679, 0.5243968538462991, 0.5261944808135561, 0.4923524285054516, 0.5253580890769276, 0.5265469799465016, 0.5237779723096413, 0.5248314659409267, 0.49737994470588537, 0.5258647004861914, 0.5223948369716719, 0.525197873968396, 0.5240052618026899, 0.49813926362251504, 0.5252998554889498, 0.5241347663415017, 0.5222958035049906, 0.5257403982552875, 0.49661078434219685, 0.5247939545114846, 0.5257963687313024, 0.5224902854922304, 0.5239797451637682, 0.5003422464844393, 0.5254318208532622, 0.5263530350767771, 0.5262280774032895, 0.5246813843108598, 0.4968952869432524, 0.5245253248621193, 0.5246131672267168, 0.525337493439658, 0.5263138958794327, 0.4972733825708236, 0.5248018723259138, 0.5254314230863131, 0.5245133350551076, 0.525670975974026, 0.4971809947765105, 0.5244207243896171, 0.5255797205593677, 0.5264314402240966, 0.5272218077364568, 0.5052044784744794, 0.5249462211954439, 0.5259541679083206, 0.5248858497356971, 0.5245939328397159, 0.5022954261121937, 0.5258595379388523, 0.5230897743619728, 0.5226367636169414, 0.5253169653019867, 0.5259918538776531, 0.524864219113456, 0.5249087259948145, 0.524017838122143, 0.5237473533251611, 0.49913299638189645, 0.5261281126984911, 0.5212131015912884, 0.5262912732810833, 0.5246300726712547, 0.5270697382727837, 0.5248463751435967, 0.5243226892342379, 0.5213714606162723, 0.5252054352018651, 0.5269694345043034, 0.5228846896038252, 0.5250753022807085, 0.5263587362130735, 0.5232794771849479, 0.525149512411686, 0.5252570560636967, 0.5261877962089956, 0.5259355998101214, 0.5247787182819239, 0.5262263243189889, 0.5261225041891702, 0.5262747225621676, 0.5259187284042898, 0.5265729089456632, 0.5250225595560285, 0.5256567382727351, 0.5267726457149635, 0.5268295454435241, 0.5260865551921488, 0.5259959866586523, 0.5259575228735759, 0.5276253046490748]}, report=None)
AUROC of tuned SVM: 0.528
Training model...
Validating model
Testing...
	Confusion matrix: 
[[131   0]
 [114   0]]
	Accuracy:
0.5346938775510204

Process finished with exit code 0


'''