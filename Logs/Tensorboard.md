# Tensorboard Logs

To visualize your logs using Tensorboard, you can use the following command:

```bash
tensorboard --logdir=.
```

This command will start a Tensorboard server that reads logs from the current directory.

## Steps to Use Tensorboard

1. Ensure you have Tensorboard installed. If not, you can install it using pip:
  ```bash
  pip install tensorboard
  ```

2. Navigate to the directory containing your logs.

3. Run the Tensorboard command:
  ```bash
  tensorboard --logdir=.
  ```

4. Open a web browser and go to the URL provided by Tensorboard (usually `http://localhost:6006`).

## Additional Resources

- [Tensorboard Documentation](https://www.tensorflow.org/tensorboard)
- [Tensorboard GitHub Repository](https://github.com/tensorflow/tensorboard)