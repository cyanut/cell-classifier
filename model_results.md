Notes on models

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
