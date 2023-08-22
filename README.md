# Deploy TF-Lite model to AWS cloud: ECR, Lambda and API Gateway

This is a classifier for dinosaur and dragon developed with AWS services.

## Pretrained Model `dino_dragon.tflite`
```
def make_model(input_size=150, learning_rate=0.002, size_inner=64):

    inputs = keras.Input(shape=(input_size, input_size, 3))

    conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(inputs)
    pool = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    flat = keras.layers.Flatten()(pool)
    FC = keras.layers.Dense(size_inner, activation='relu')(flat)
    outputs = keras.layers.Dense(1, activation='sigmoid')(FC)

    model = keras.Model(inputs, outputs)
    
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.8)
    loss = tf.keras.losses.BinaryCrossentropy()

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model
```


## Testing with `testing.py`
- Ensure the host is correct
- Change the `img_url` to desired image in the script
```
python test.py
```

## To Run this APP on your own
1. Train a Tensorflow model and clone this repo
2. Convert your TF model to TF-Lite with `convert.py`
3. Define the lambda function 
    - See `lambda_function.py`
    - Test with terminal 
    ```
    python
    >>> import lambda_function
    >>> event = {"url": "{some image url}"}
    >>> lambda_function.lambda_handler(event, None)
    ```
4. Build the image 
    - See `Dockerfile`
    - Run and test
    ```
    docker build -t {image_name} .
    docker run --rm -p 8080:8080 {image_name}
    python test.py
    ```
5. Publishing to AWS
    - Publish image to ECR
    ```
    aws ecr create-repository --repository-name {repo_name}
    aws ecr get-login-password --region {region} | docker login\
        --username AWS \
        --password-stdin {aws_account_id}.dkr.ecr.{region}.amazonaws.com
    docker tag {docker image} {repositoryUri}:{Tag}
    docker push {repositoryUri}:{Tag}
    ```
    - Create a lambda function using the ECR image
        - Change configuration (*needed for model initialization*)
            - timeout -> 30s
            - memory -> 1024ME
    - Expose lambda function to API Gateway
        - Create resource
        - Create method
        - Deploy API

