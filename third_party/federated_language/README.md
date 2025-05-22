# Federated Language Patches

1.  Clone the repository.

    ```shell
    $ git clone https://github.com/google-parfait/federated-language.git "/tmp/federated-language"
    $ cd "/tmp/federated-language"
    ```

1.  Checkout the commit.

    ```shell
    $ git checkout <COMMIT>
    ```

1.  Create the `python_deps` patch.

    Make the changes.

    ```shell
    $ buildozer 'remove deps @federated_language_pypi//absl_py' //...:*
    $ buildozer 'remove deps @federated_language_pypi//attrs' //...:*
    $ buildozer 'remove deps @federated_language_pypi//dm_tree' //...:*
    $ buildozer 'remove deps @federated_language_pypi//ml_dtypes' //...:*
    $ buildozer 'remove deps @federated_language_pypi//numpy' //...:*
    $ buildozer 'remove deps @federated_language_pypi//typing_extensions' //...:*
    ```

    Confirm no more changes are required.

    ```shell
    $ find "." -type f -print0 | xargs -0 grep "@federated_language_pypi"
    ```

    Create the patch.

    ```shell
    $ git diff --no-prefix \
        > "<CLIENT>/tensorflow_federated/third_party/federated_language/python_deps.patch"
    ```

1.  Create the `structure_visibility` patch.

    Note: This patch requires the previous patch to be applied.

    Stage the changes from the previous step.

    ```shell
    $ git add .
    ```

    Make the changes.

    ```shell
    $ buildozer 'add visibility //visibility:public' //federated_language/common_libs:structure
    ```

    Create the patch.

    ```shell
    $ git diff --no-prefix \
        > "<CLIENT>/tensorflow_federated/third_party/federated_language/structure_visibility.patch"
    ```
