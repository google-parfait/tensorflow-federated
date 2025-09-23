package org.jetbrains.tff.engine;

import java.io.FileOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

public class AggregationTool {
    public static void main(String[] args) throws Exception {
        if (args.length < 3) {
            System.err.println("Usage: java AggregationTool <plan_bin> <checkpoint1> [<checkpoint2> ...] <output_checkpoint>");
            System.exit(1);
        }

        String buildWorkingDir = System.getenv("BUILD_WORKING_DIRECTORY");
        Path baseDir = (buildWorkingDir != null)
            ? Paths.get(buildWorkingDir)
            : Paths.get(System.getProperty("user.dir"));

        var planPath = baseDir.resolve(args[0]).normalize();
        var outputCheckpointPath = baseDir.resolve(args[args.length - 1]).normalize();
        List<String> checkpointPaths = java.util.Arrays.asList(args).subList(1, args.length - 1);

        byte[] planBytes = Files.readAllBytes(planPath);
        byte[][] checkpoints = new byte[checkpointPaths.size()][];
        for (int i = 0; i < checkpointPaths.size(); i++) {
            var checkpointPath = baseDir.resolve(checkpointPaths.get(i)).normalize();
            checkpoints[i] = Files.readAllBytes(checkpointPath);
        }

        // Create and run AggregationSession
        AggregationSession session = AggregationSession.createFromByteArray(planBytes);
        session.accumulate(checkpoints);
        byte[] aggregatedCheckpoint = session.report();

        // Write output
        try (FileOutputStream out = new FileOutputStream(outputCheckpointPath.toFile())) {
            out.write(aggregatedCheckpoint);
        }

        System.out.println("Aggregated checkpoint written to: " + outputCheckpointPath);
    }
}
