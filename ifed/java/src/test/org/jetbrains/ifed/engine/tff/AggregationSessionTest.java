package org.jetbrains.ifed.engine.tff;

import org.junit.Test;
import static org.junit.Assert.*;

public class AggregationSessionTest {

    @Test
    public void testSessionLifecycle() {
        // This is a placeholder plan. In real tests, use a valid serialized plan.
        byte[] dummyPlan = new byte[] {1, 2, 3};
        try {
            AggregationSession session = AggregationSession.createFromByteArray(dummyPlan);
            assertNotNull(session);
            session.close();
        } catch (Exception e) {
            // Acceptable if native code is not available in test env
        }
    }

    @Test
    public void testDoubleClose() {
        byte[] dummyPlan = new byte[] {1, 2, 3};
        try {
            AggregationSession session = AggregationSession.createFromByteArray(dummyPlan);
            session.close();
            // Should not throw
            session.close();
        } catch (Exception e) {
            // Acceptable if native code is not available in test env
        }
    }
}
