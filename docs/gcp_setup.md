# Setup simulations with TFF on GCP

This tutorial will describe how to do the steps required for the setup
high-performance simulations with TFF on GCP.

1.  [Install and initialize the Cloud SDK.](https://cloud.google.com/sdk/docs/quickstarts).

1.  Run a simulation on a single runtime container.

    1.  Start a single runtime container.

        1.  [Create a Compute Engine instance](https://cloud.google.com/endpoints/docs/grpc/get-started-compute-engine-docker#create_vm).

            NOTE: To have Docker pre-installed, selected a "Container-Optimized"
            boot disk when starting the VM.

        1.  `ssh` into the instance.

            ```shell
            $ gcloud compute ssh <instance>
            ```

        1.  Run the runtime container in the background.

            ```shell
            $ docker run \
                --detach \
                --name=runtime \
                --publish=8000:8000 \
                gcr.io/tensorflow-federated/runtime
            ```

        1.  Exit the instance.

            ```shell
            $ exit
            ```

        1.  Get the internal IP address of the instance.

            This is used later as a parameter to our test script.

            ```shell
            $ gcloud compute instances describe <instance> \
                --format='get(networkInterfaces[0].networkIP)'
            ```

    1.  Start and run a simulation on a client container.

        1.  [Create a Compute Engine instance](https://cloud.google.com/endpoints/docs/grpc/get-started-compute-engine-docker#create_vm).

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
                --name=client \
                --volume ~:/simulation \
                --workdir /simulation \
                gcr.io/tensorflow-federated/client \
                bash
            ```

        1.  Run the Python script.

            Using the internal IP address of the instance running the runtime
            container.

            ```shell
            $ python3 test.py --host '<internal IP address>'
            ```

        1.  Exit the container.

            ```shell
            $ exit
            ```

        1.  Exit the instance.

            ```shell
            $ exit
            ```

1.  Configure runtime and client images from source.

    If you wanted to run a simulation using TFF from source instead of a
    released version of TFF, you need to: build the runtime and client images
    from source; publish those images to your own container registry; and
    finally create the runtime and client containers using those images instead
    of the released images that we provide. See the
    [Container Registry documentation](https://cloud.google.com/container-registry/docs/)
    for more information.

    1.  Configure a runtime image.

        ```shell
        $ bazel run //tensorflow_federated/tools/runtime/gcp:build_image
        $ bazel run //tensorflow_federated/tools/runtime/gcp:publish_image -- \
            <runtime registry>
        ```

    1.  Configure a client image.

        ```shell
        $ bazel run //tensorflow_federated/tools/client:build_image
        $ bazel run //tensorflow_federated/tools/client:publish_image -- \
            <client registry>
        ```
