Notes on models

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
