<b>SuperResolution using an Efficient Sub-Pixel Convolutional Neural Network</b>

Model (model_epoch_1000.pth) has been trained and can be executed directly on Low Resolution images to upscale and/or enhance its resolution using below command -

python predict.py --input_image <input_image>.png --model model_epoch_1000.pth --output_filename <output_image>.png

For example -

<I>python predict.py --input_image property1_input.png --model model_epoch_1000.pth --output_filename property1_output.jpg</I>
 



<b>Alternatively, a new model can be trained using below steps -</b>

Following are the Instructions to train a new model  - 

python main.py --upscale_factor 2 --batchSize 1 --testBatchSize 50 --nEpochs 100 --lr 0.001


Following are the Instructions to test a trained model on new images - 

python predict.py --input_image <input_image>.jpg --model model_epoch_1000.pth --output_filename <output_image>.jpg




<b> Results </b>

Please check out "Results" folder to see some of the quantarium and real world images that has been enhanced using techniques depicted in this project -




<b> TBD </b>

More work is needed to enhance the image quality further which require additional effort in regards to hyper-parameters modifications and more training.


<b>Reference:-</b>

https://arxiv.org/pdf/1609.05158.pdf
