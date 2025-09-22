package org.jetbrains.tff.engine;

import java.io.FileOutputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

public class AggregationSessionTool {
    public static void main(String[] args) throws Exception {
        if (args.length < 3) {
            System.err.println("Usage: java AggregationSessionTool <plan_bin> <checkpoint1> [<checkpoint2> ...] <output_checkpoint>");
            System.exit(1);
        }

        String planPath = args[0];
        String outputCheckpoint = args[args.length - 1];
        List<String> checkpointPaths = java.util.Arrays.asList(args).subList(1, args.length - 1);

        byte[] planBytes = Files.readAllBytes(Paths.get(planPath));
        byte[][] checkpoints = new byte[checkpointPaths.size()][];
        for (int i = 0; i < checkpointPaths.size(); i++) {
            checkpoints[i] = Files.readAllBytes(Paths.get(checkpointPaths.get(i)));
        }

        // Create and run AggregationSession
        AggregationSession session = AggregationSession.createFromByteArray(planBytes);
        session.accumulate(checkpoints);
        byte[] aggregatedCheckpoint = session.report();

        // Write output
        try (FileOutputStream out = new FileOutputStream(outputCheckpoint)) {
            out.write(aggregatedCheckpoint);
        }
        System.out.println("Aggregated checkpoint written to: " + outputCheckpoint);
    }
}
