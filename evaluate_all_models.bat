@echo off
REM Batch script to evaluate all models across all classifications
REM Run this from the pytorch-classification directory

echo ====================================
echo Starting Model Evaluation
echo ====================================
echo.

REM A1_classification models (5 models)
echo [1/30] Evaluating A1_classification - efficientnet_b0...
python modelseval.py -d "D:\Projects\Research\new-project\data\classification_training\A1_classification_training" --arch efficientnet_b0 --checkpoint "checkpoints\my_model\A1_classification\efficientnet_b0"
echo.

echo [2/30] Evaluating A1_classification - resnet18...
python modelseval.py -d "D:\Projects\Research\new-project\data\classification_training\A1_classification_training" --arch resnet18 --checkpoint "checkpoints\my_model\A1_classification\resnet18"
echo.

echo [3/30] Evaluating A1_classification - resnet50...
python modelseval.py -d "D:\Projects\Research\new-project\data\classification_training\A1_classification_training" --arch resnet50 --checkpoint "checkpoints\my_model\A1_classification\resnet50"
echo.

echo [4/30] Evaluating A1_classification - vgg16...
python modelseval.py -d "D:\Projects\Research\new-project\data\classification_training\A1_classification_training" --arch vgg16 --checkpoint "checkpoints\my_model\A1_classification\vgg16"
echo.

echo [5/30] Evaluating A1_classification - vit_b_16...
python modelseval.py -d "D:\Projects\Research\new-project\data\classification_training\A1_classification_training" --arch vit_b_16 --checkpoint "checkpoints\my_model\A1_classification\vit_b_16"
echo.

REM A2_classification models (5 models)
echo [6/30] Evaluating A2_classification - efficientnet_b0...
python modelseval.py -d "D:\Projects\Research\new-project\data\classification_training\A2_classification_training" --arch efficientnet_b0 --checkpoint "checkpoints\my_model\A2_classification\efficientnet_b0"
echo.

echo [7/30] Evaluating A2_classification - resnet18...
python modelseval.py -d "D:\Projects\Research\new-project\data\classification_training\A2_classification_training" --arch resnet18 --checkpoint "checkpoints\my_model\A2_classification\resnet18"
echo.

echo [8/30] Evaluating A2_classification - resnet50...
python modelseval.py -d "D:\Projects\Research\new-project\data\classification_training\A2_classification_training" --arch resnet50 --checkpoint "checkpoints\my_model\A2_classification\resnet50"
echo.

echo [9/30] Evaluating A2_classification - vgg16...
python modelseval.py -d "D:\Projects\Research\new-project\data\classification_training\A2_classification_training" --arch vgg16 --checkpoint "checkpoints\my_model\A2_classification\vgg16"
echo.

echo [10/30] Evaluating A2_classification - vit_b_16...
python modelseval.py -d "D:\Projects\Research\new-project\data\classification_training\A2_classification_training" --arch vit_b_16 --checkpoint "checkpoints\my_model\A2_classification\vit_b_16"
echo.

REM A3_classification models (5 models)
echo [11/30] Evaluating A3_classification - efficientnet_b0...
python modelseval.py -d "D:\Projects\Research\new-project\data\classification_training\A3_classification_training" --arch efficientnet_b0 --checkpoint "checkpoints\my_model\A3_classification\efficientnet_b0"
echo.

echo [12/30] Evaluating A3_classification - resnet18...
python modelseval.py -d "D:\Projects\Research\new-project\data\classification_training\A3_classification_training" --arch resnet18 --checkpoint "checkpoints\my_model\A3_classification\resnet18"
echo.

echo [13/30] Evaluating A3_classification - resnet50...
python modelseval.py -d "D:\Projects\Research\new-project\data\classification_training\A3_classification_training" --arch resnet50 --checkpoint "checkpoints\my_model\A3_classification\resnet50"
echo.

echo [14/30] Evaluating A3_classification - vgg16...
python modelseval.py -d "D:\Projects\Research\new-project\data\classification_training\A3_classification_training" --arch vgg16 --checkpoint "checkpoints\my_model\A3_classification\vgg16"
echo.

echo [15/30] Evaluating A3_classification - vit_b_16...
python modelseval.py -d "D:\Projects\Research\new-project\data\classification_training\A3_classification_training" --arch vit_b_16 --checkpoint "checkpoints\my_model\A3_classification\vit_b_16"
echo.

REM B1_classification models (5 models)
echo [16/30] Evaluating B1_classification - efficientnet_b0...
python modelseval.py -d "D:\Projects\Research\new-project\data\classification_training\B1_classification_training" --arch efficientnet_b0 --checkpoint "checkpoints\my_model\B1_classification\efficientnet_b0"
echo.

echo [17/30] Evaluating B1_classification - resnet18...
python modelseval.py -d "D:\Projects\Research\new-project\data\classification_training\B1_classification_training" --arch resnet18 --checkpoint "checkpoints\my_model\B1_classification\resnet18"
echo.

echo [18/30] Evaluating B1_classification - resnet50...
python modelseval.py -d "D:\Projects\Research\new-project\data\classification_training\B1_classification_training" --arch resnet50 --checkpoint "checkpoints\my_model\B1_classification\resnet50"
echo.

echo [19/30] Evaluating B1_classification - vgg16...
python modelseval.py -d "D:\Projects\Research\new-project\data\classification_training\B1_classification_training" --arch vgg16 --checkpoint "checkpoints\my_model\B1_classification\vgg16"
echo.

echo [20/30] Evaluating B1_classification - vit_b_16...
python modelseval.py -d "D:\Projects\Research\new-project\data\classification_training\B1_classification_training" --arch vit_b_16 --checkpoint "checkpoints\my_model\B1_classification\vit_b_16"
echo.

REM B2_classification models (5 models)
echo [21/30] Evaluating B2_classification - efficientnet_b0...
python modelseval.py -d "D:\Projects\Research\new-project\data\classification_training\B2_classification_training" --arch efficientnet_b0 --checkpoint "checkpoints\my_model\B2_classification\efficientnet_b0"
echo.

echo [22/30] Evaluating B2_classification - resnet18...
python modelseval.py -d "D:\Projects\Research\new-project\data\classification_training\B2_classification_training" --arch resnet18 --checkpoint "checkpoints\my_model\B2_classification\resnet18"
echo.

echo [23/30] Evaluating B2_classification - resnet50...
python modelseval.py -d "D:\Projects\Research\new-project\data\classification_training\B2_classification_training" --arch resnet50 --checkpoint "checkpoints\my_model\B2_classification\resnet50"
echo.

echo [24/30] Evaluating B2_classification - vgg16...
python modelseval.py -d "D:\Projects\Research\new-project\data\classification_training\B2_classification_training" --arch vgg16 --checkpoint "checkpoints\my_model\B2_classification\vgg16"
echo.

echo [25/30] Evaluating B2_classification - vit_b_16...
python modelseval.py -d "D:\Projects\Research\new-project\data\classification_training\B2_classification_training" --arch vit_b_16 --checkpoint "checkpoints\my_model\B2_classification\vit_b_16"
echo.

REM B3_classification models (5 models)
echo [26/30] Evaluating B3_classification - efficientnet_b0...
python modelseval.py -d "D:\Projects\Research\new-project\data\classification_training\B3_classification_training" --arch efficientnet_b0 --checkpoint "checkpoints\my_model\B3_classification\efficientnet_b0"
echo.

echo [27/30] Evaluating B3_classification - resnet18...
python modelseval.py -d "D:\Projects\Research\new-project\data\classification_training\B3_classification_training" --arch resnet18 --checkpoint "checkpoints\my_model\B3_classification\resnet18"
echo.

echo [28/30] Evaluating B3_classification - resnet50...
python modelseval.py -d "D:\Projects\Research\new-project\data\classification_training\B3_classification_training" --arch resnet50 --checkpoint "checkpoints\my_model\B3_classification\resnet50"
echo.

echo [29/30] Evaluating B3_classification - vgg16...
python modelseval.py -d "D:\Projects\Research\new-project\data\classification_training\B3_classification_training" --arch vgg16 --checkpoint "checkpoints\my_model\B3_classification\vgg16"
echo.

echo [30/30] Evaluating B3_classification - vit_b_16...
python modelseval.py -d "D:\Projects\Research\new-project\data\classification_training\B3_classification_training" --arch vit_b_16 --checkpoint "checkpoints\my_model\B3_classification\vit_b_16"
echo.

echo ====================================
echo All Model Evaluations Complete!
echo ====================================
echo.
echo Total Models Evaluated: 30 (5 models x 6 classifications)
echo.
echo Results saved in respective checkpoint folders:
echo - evaluation_metrics.txt
echo - confusion_matrix.csv
echo - confusion_matrix.png
echo - class_names.txt
echo.
pause