Notes on models

`./cell_classifier.py --njobs 4 -o model_20160520_gt8_rmexcluded_gauss_f1score.pkl -x 14 15 16 17 18 19 20 21 22 -y 23 --benchmark 13 ../ccell-data/brain-map/algorithmdev/cropsections_joineddetails_20160520_gt8_rmexcluded_gauss.csv`
[('knn', 0.80028668384209678), ('gsvm', 0.80667518729722321), ('rf', 0.78498380620119923), ('gnb', 0.77152816124512835), ('lda', 0.79653964178078129), ('voting', 0.80466154756517572)]
('LDA coef:', [('meani', 0.21617623891435017), ('equivdiameter', 2.8937220222490243), ('circularity', -0.08681813478225732), ('eccentricity', -0.19410876132473107), ('area', -2.0570416275254964), ('minor_axis_length', -0.20137845778309882), ('major_axis_length', 0.25765311007233194), ('min_intensity', -0.36483983599353464), ('max_intensity', 0.28480908171778063)])
Best model is: gsvm, with parameters {'kernel': 'rbf', 'C': 1.0, 'gamma': 0.1111111111111111}
Validation F1 Score: 0.806675187297
Testing precision: 0.733417561592, recall: 0.946210268949, F1 Score: 0.826334519573, support: None
Benchmark precision: 0.60736196319, recall: 0.080684596577, F1 score: 0.142446043165, support: None
Pickled the best model in model_20160520_gt8_rmexcluded_gauss_f1score.pkl
target: ['ground_truth']
features tried ['meani', 'equivdiameter', 'circularity', 'eccentricity', 'area', 'minor_axis_length', 'major_axis_length', 'min_intensity', 'max_intensity']

`./cell_classifier.py -o model_20160519_gt8_rmexcluded_f1score.pkl -x 14 15 16 17 18 19 20 21 22 -y 23 --benchmark 13 ../ccell-data/brain-map/algorithmdev/cropsections_joineddetails_20160519_gt8_rmexcluded.csv`
[('knn', 0.79848887187092588), ('gsvm', 0.80541524092321271), ('rf', 0.78711567323117615), ('gnb', 0.79767486646639341), ('lda', 0.79885051256243733), ('voting', 0.80325688188278088)]
('LDA coef:', [('meani', 0.68193218885288232), ('equivdiameter', 14.600604215736112), ('circularity', 0.17165900594298547), ('eccentricity', -0.88299242099644859), ('area', -5.0562179198635242), ('minor_axis_length', -3.7081823921458383), ('major_axis_length', -1.8884962966714163), ('min_intensity', -0.87427120345366127), ('max_intensity', 0.29504017196907162)])
Best model is: gsvm, with parameters {'kernel': 'rbf', 'C': 27.0, 'gamma': 0.1111111111111111}
Validation F1 Score: 0.805415240923
Testing precision: 0.706470588235, recall: 0.940485512921, F1 Score: 0.80685253611, support: None
Benchmark precision: 0.693577981651, recall: 0.296006264683, F1 score: 0.414928649835, support: None
Pickled the best model in model_20160519_gt8_rmexcluded_f1score.pkl
target: ['ground_truth']
features tried ['meani', 'equivdiameter', 'circularity', 'eccentricity', 'area', 'minor_axis_length', 'major_axis_length', 'min_intensity', 'max_intensity']

`./cell_classifier.py -o model_20160519_gt8_f1score.pkl -x 14 15 16 17 18 19 20 21 22 -y 23 --benchmark 13 ../ccell-data/brain-map/algorithmdev/cropsections_joineddetails_20160518_gt8.csv`
Best model is: gsvm, with parameters {'kernel': 'rbf', 'C': 0.33333333333333331, 'gamma': 1.0}
Validation F1 Score: 0.81977625301
Testing precision: 0.727702303603, recall: 0.935459377373, F1 Score: 0.818604651163, support: None
Benchmark precision: 0.717557251908, recall: 0.285497342445, F1 score: 0.408473655622, support: None
Pickled the best model in model_20160519_gt8_f1score.pkl
target: ['ground_truth']
features tried ['meani', 'equivdiameter', 'circularity', 'eccentricity', 'area', 'minor_axis_length', 'major_axis_length', 'min_intensity', 'max_intensity']


`./cell_classifier.py -x 14 15 16 17 18 19 20 21 22 -y 23 --benchmark 13 ../ccell/brain-map/algorithmdev/cropsections_joineddetails_20160329_gt8_bboxcriteria_lesszero.csv`
Best model is: RandomForest, with parameters {'max_features': 1, 'n_estimators': 27}
Validation F1 Score: 0.972797103886
Testing F1 Score: 0.975899147201
Benchmark F1 Score: 0.503340757238
target: ['ground_truth']
features tried ['meani', 'equivdiameter', 'circularity', 'eccentricity', 'area', 'minor_axis_length', 'major_axis_length', 'min_intensity', 'max_intensity']

`./cell_classifier.py -x 14 15 16 17 18 19 20 21 22 -y 23 -o model_20160331_gt20_bbox_lesszero_20160331_precision.pkl --score precision ../ccell/brain-map/algorithmdev/cropsections_joineddetails_20160331_gt20_bboxcriteria_lesszero.csv`
Best model is: RandomForest, with parameters {'max_features': 8, 'n_estimators': 3}
Validation F1 Score: 0.96397316
Testing F1 Score: 0.974989749897
Pickled the best model in model_20160331_gt20_bbox_lesszero_20160331_precision.pkl

`./cell_classifier.py -x 14 15 16 17 18 19 20 21 22 -y 23 -o model_20160331_gt20_bbox_lesszero_20160331.pkl ../ccell/brain-map/algorithmdev/cropsections_joineddetails_20160331_gt20_bboxcriteria_lesszero.csv`
Best model is: RandomForest, with parameters {'max_features': 2, 'n_estimators': 243}
Validation F1 Score: 0.980621106746
Testing F1 Score: 0.984590429846
Pickled the best model in model_20160331_gt20_bbox_lesszero_20160331.pkl

`./cell_classifier.py -x 14 15 16 17 18 19 20 21 22 -y 23 -o model_20160331_gt14_bbox_lesszero_20160331.pkl ../ccell/brain-map/algorithmdev/cropsections_joineddetails_20160331_gt14_bboxcriteria_lesszero.csv`
Best model is: RandomForest, with parameters {'max_features': 1, 'n_estimators': 243}
Validation F1 Score: 0.978451467081
Testing F1 Score: 0.976127320955
Pickled the best model in model_20160331_gt14_bbox_lesszero_20160331.pkl

`./cell_classifier.py -x 14 15 16 17 18 19 20 21 22 -y 23 -o model_20160329_gt8_bbox_lesszero_20160329_withnormalizer.pkl ../ccell/brain-map/algorithmdev/cropsections_joineddetails_20160329_gt8_bboxcriteria_lesszero.csv`
Best model is: RandomForest, with parameters {'max_features': 1, 'n_estimators': 27}
Validation F1 Score: 0.972688059309
Testing F1 Score: 0.975501113586
Pickled the best model in model_20160329_gt8_bbox_lesszero_20160329_withnormalizer.pkl

# NOTE
ABOVE RAN AFTER NORMALIZER ADDED TO MODEL (before it was outside of the model)

`./cell_classifier.py -x 14 15 16 17 18 19 20 21 22 -y 23 -o model_20160329_gt8_bbox_lesszero_20160329.pkl ../ccell/brain-map/algorithmdev/cropsections_joineddetails_20160329_gt8_bboxcriteria_lesszero.csv`
Best model is: RandomForest, with parameters {'max_features': 1, 'n_estimators': 243}
Validation F1 Score: 0.972681564303
Testing F1 Score: 0.975501113586
Pickled the best model in model_20160329_gt8_bbox_lesszero_20160329.pkl

`./cell_classifier.py -x 14 15 16 17 18 19 20 21 22 -y 23 -o model_gt10_f1_bbox_morezero.pkl ../ccell/brain-map/algorithmdev/cropsections_joineddetails_20160322_gt10_bboxcriteria_morezero.csv`
Best model is: GaussianSVM, with parameters {'kernel': 'rbf', 'C': 81.0, 'gamma': 0.00411522633744856}
Validation F1 Score: 0.736872661669
Testing F1 Score: 0.742482652274
Pickled the best model in model_gt10_f1_bbox_morezero.pkl

`./cell_classifier.py -x 14 15 16 17 18 19 20 21 22 -y 23 -o model_gt4_f1_bbox_morezero.pkl ../ccell/brain-map/algorithmdev/cropsections_joineddetails_20160321_gt4_bboxcriteria_morezero.csv`
Best model is: GaussianSVM, with parameters {'kernel': 'rbf', 'C': 9.0, 'gamma': 0.037037037037037035}
Validation F1 Score: 0.724603978948
Testing F1 Score: 0.732394366197
Pickled the best model in model_gt4_f1_bbox_morezero.pkl

`/cell_classifier.py -x 14 15 16 17 18 19 20 21 22 -y 23 --score f1_weighted -o model_gt4_f1weighted_bbox_morezero.pkl --image ../ccell/sampledata/colm-arctdt/jon-test20150906_0150.tif ../ccell/brain-map/algorithmdev/cropsections_joineddetails_20160321_gt4_bboxcriteria_morezero.csv`
Best model is: GaussianSVM, with parameters {'kernel': 'rbf', 'C': 9.0, 'gamma': 0.037037037037037035}
Validation F1 Score: 0.942701325325
Testing F1 Score: 0.732394366197
Pickled the best model in model_gt4_f1weighted_bbox_morezero.pkl

`./cell_classifier.py -x 14 15 16 17 18 19 20 21 22 -y 23 -o model_gt4_precision_bbox_morezero.pkl ../ccell/brain-map/algorithmdev/cropsections_joineddetails_20160321_gt4_bboxcriteria_morezero.csv`
Best model is: GaussianSVM, with parameters {'kernel': 'rbf', 'C': 0.00411522633744856, 'gamma': 0.012345679012345678}
Validation F1 Score: 0.787310519664
Testing F1 Score: 0.654939106901
Pickled the best model in model_gt4_precision_bbox_morezero.pkl

`./cell_classifier.py -x 14 15 16 17 18 19 20 21 22 -y 23 -o model_gt4_f1_radius_morezero.pkl ../ccell/brain-map/algorithmdev/cropsections_joineddetails_20160321_gt4_radiuscriteria_morezero.csv`
Best model is: GaussianSVM, with parameters {'kernel': 'rbf', 'C': 9.0, 'gamma': 0.037037037037037035}
Validation F1 Score: 0.68011703992
Testing F1 Score: 0.682213077275
Pickled the best model in model_gt4_f1_radius_morezero.pkl

`./cell_classifier.py -x 14 15 16 17 18 19 20 21 22 -y 23 --score precision -o model_gt4_precision_radius_morezero.pkl ../ccell/brain-map/algorithmdev/cropsections_joineddetails_20160321_gt4_radiuscriteria_morezero.csv`
Best model is: GaussianSVM, with parameters {'kernel': 'rbf', 'C': 1.0, 'gamma': 81.0}
Validation F1 Score: 0.804156414988
Testing F1 Score: 0.0257116620753
Pickled the best model in model_gt4_precision_radius_morezero.pkl
