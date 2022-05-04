# TFF simulations on GCP

This tutorial will describe how to run TFF simulations on GCP.

## Run a simulation on a single runtime container

### 1. [Install and initialize the Cloud SDK.](https://cloud.google.com/sdk/docs/quickstarts).

### 2. Clone the TensorFlow Federated repository.

```shell
$ git clone https://github.com/tensorflow/federated.git
$ cd "federated"
```

### 3. Run a single runtime container.

1.  Build a runtime container.

    ```shell
    $ docker build \
        --network=host \
        --tag "<registry>/tff-runtime" \
        --file "tensorflow_federated/tools/runtime/container/Dockerfile" \
        .
    ```

1.  Publish the runtime container.

    ```shell
    $ docker push <registry>/tff-runtime
    ```

1.  Create a Compute Engine instance

    1.  In the Cloud Console, go to the
        [VM Instances](https://console.cloud.google.com/compute/instances) page.

    1.  Click **Create instance**.

    1.  In the **Firewall** section, select **Allow HTTP traffic** and **Allow
        HTTPS traffic**.

    1.  Click **Create** to create the instance.

1.  `ssh` into the instance.

    ```shell
    $ gcloud compute ssh <instance>
    ```

1.  Run the runtime container in the background.

    ```shell
    $ docker run \
        --detach \
        --name=tff-runtime \
        --publish=8000:8000 \
        <registry>/tff-runtime
    ```

1.  Exit the instance.

    ```shell
    $ exit
    ```

1.  Get the internal **IP address** of the instance.

    This is used later as a parameter to our test script.

    ```shell
    $ gcloud compute instances describe <instance> \
        --format='get(networkInterfaces[0].networkIP)'
    ```

### 4. Run a simulation on a client container.

1.  Build a client container.

    ```shell
    $ docker build \
        --network=host \
        --tag "<registry>/tff-client" \
        --file "tensorflow_federated/tools/client/latest.Dockerfile" \
        .
    ```

1.  Publish the client container.

    ```shell
    $ docker push <registry>/tff-client
    ```

1.  Create a Compute Engine instance

    1.  In the Cloud Console, go to the
        [VM Instances](https://console.cloud.google.com/compute/instances) page.

    1.  Click **Create instance**.

    1.  In the **Firewall** section, select **Allow HTTP traffic** and **Allow
        HTTPS traffic**.

    1.  Click **Create** to create the instance.

1.  Copy your experiement to the Compute Engine instance.

    ```shell
    $ gcloud compute scp \
        "tensorflow_federated/tools/client/test.py" \
        <instance>:~
    ```

1.  `ssh` into the instance.

    ```shell
    $ gcloud compute ssh <instance>
    ```

1.  Run the client container interactively.

    The string "Hello World" should print to the terminal.

    ```shell
    $ docker run \
        --interactive \
        --tty \
        --name=tff-client \
        --volume ~/:/simulation \
        --workdir /simulation \
        <registry>/tff-client \
        bash
    ```

1.  Run the Python script.

    Using the internal **IP address** of the instance running the runtime
    container.

    ```shell
    $ python3 test.py --host '<IP address>'
    ```

1.  Exit the container.

    ```shell
    $ exit
    ```

1.  Exit the instance.

    ```shell
    $ exit
    ```
