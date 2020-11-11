package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files([
    "building_your_own_federated_learning_algorithm.ipynb",
    "custom_federated_algorithms_1.ipynb",
    "custom_federated_algorithms_2.ipynb",
    "federated_learning_for_image_classification.ipynb",
    "federated_learning_for_text_generation.ipynb",
    "simulations.ipynb",
    "tff_for_federated_learning_research_compression.ipynb",
])

# This exists for the API doc generator. The doc generator tool looks for API
# usage in tutorial content and links to the tutorials in the API doc content.
filegroup(
    name = "tutorials",
    srcs = glob(["**"]),
)
