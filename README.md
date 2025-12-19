# BadukDeepLearningModel (9x9 Go AI)

A Deep Learning model trained to play the game of Go (Baduk) on a 9x9 board. 

I quantized and deployed the best run I had on my website. You can play against the model here:
 **[denizay.github.io/blog/baduk.html](https://denizay.github.io/blog/baduk.html)**

## Architecture
* **Backbone:** Residual CNN Blocks with 3x3 kernels. 96 features per kernel and 18 blocks in the best run.
* **Heads:** Policy Head only (no Value Head).
    * The policy head reduces the board to `2x9x9` features before flattening.
    * Outputs 82 probabilities (81 board moves + 1 pass).
* **Training:** Different RL schedulers and optimizers ara available. Trains with mixed precision, `bfloat16` if available else `float16`.
* **Deployment:** Quantized to `int8` using dynamic quantization for a 75% size reduction (12MB &rarr; 3.2MB) with negligible accuracy loss.

## Dataset & Training
The model was trained on high-rank games from the **GoQuest** server.
* **Dataset Source:** [Computer-Go Archive](https://www.eugeneweb.com/pipermail/computer-go/2015-December/008353.html)
* **Training Logs:** You can view the training metrics on the [WandB Report](https://wandb.ai/denizay333-gena/BadukDeepLearning-CNN-BiggerDataset/reports/9x9-Go-AI-Training-Runs--VmlldzoxNTQxNjU4NA?accessToken=aofm1cmc0sw4xuezr5h46533qmqcy3lvwgurju4r1s9e5f8imvrq7t4j20luwwgx).

## Usage

### 1. Prepare the Data
First, you need a collection of SGF files (e.g., from the [Computer-Go Archive](https://www.eugeneweb.com/pipermail/computer-go/2015-December/008353.html)).

Set the  ```SGF_FOLDER_PATH``` variable in create_data script to the path of the folder containing your `.sgf` files.

```bash
python3 data/create_data.py
```

### 2. Train the Model
Run the training script to parse the data and start the training loop:
```bash
python3 train.py
```

### 3. Test the Model (CLI)
Once you have trained a model, you can play against it directly in your terminal to test its performance:
```bash
python3 versus.py --model_path path/to/your/checkpoint.pth
```