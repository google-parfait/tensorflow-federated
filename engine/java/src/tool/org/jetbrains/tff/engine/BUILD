
load("@rules_java//java:java_binary.bzl", "java_binary")

java_binary(
	name = "agg_tool",
	srcs = ["AggregationTool.java"],
	main_class = "org.jetbrains.tff.engine.AggregationTool",
	deps = [
		"//engine/java/src/main/org/jetbrains/tff/engine",
	],
)

