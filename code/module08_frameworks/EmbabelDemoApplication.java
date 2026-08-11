/*
 * Embabel Goal-Directed Task Decomposition Agent (Java)
 * ======================================================
 *
 * WHAT THIS SHOWS (capabilities unique to Embabel):
 *   1. Goal-directed agents  — declare WHAT the agent should achieve, not HOW
 *   2. @Action annotations   — Spring-managed steps with condition guards
 *   3. Automatic planning    — Embabel picks the action sequence to reach a goal
 *   4. Spring DI integration — services are injectable beans, fully testable
 *   5. Domain-typed state    — immutable objects replace raw dicts
 *
 * WHY TASK DECOMPOSITION (not weather/pincode like earlier modules):
 *   - module01–03 already demonstrate weather/pincode tool calls
 *   - Task decomposition shows a PLANNING scenario where the order of
 *     actions matters and guards are essential — Embabel's real strength
 *   - The domain has clear preconditions: you can't estimate effort before
 *     decomposing, and you can't validate before estimating — perfect for
 *     Embabel's goal-directed execution model
 *
 * AGENT BEHAVIOUR:
 *   Input: a vague task description ("Build a REST API for a blog")
 *   Actions:
 *     1. decompose  — LLM breaks task into subtasks (fires when subtasks empty)
 *     2. estimate   — LLM assigns story points     (fires when subtasks set, effort unknown)
 *     3. validate   — rule-based sanity check      (fires when effort set, not yet validated)
 *     4. summarise  — LLM writes final plan        (fires when validated)
 *   Goal: summary is populated
 *
 * COMPARE WITH:
 *   - module01_raw/1.3_tool_single  : you write if/elif routing manually
 *   - module03_langchain/3.2        : LangChain agent loop, still imperative
 *   - 8.2_dspy_optimized_agent.py   : DSPy optimises prompts, you define pipeline
 *   - This file                     : declare goal + actions, Embabel plans
 *
 * PREREQUISITES:
 *   - JDK 21+, Maven or Gradle
 *   - pom.xml dependency:
 *       <dependency>
 *           <groupId>com.embabel</groupId>
 *           <artifactId>embabel-agent-api</artifactId>
 *           <version>0.5.0</version>
 *       </dependency>
 *   - Ollama-compatible endpoint
 *
 * RUN: mvn spring-boot:run
 *
 * NOTE: Java/Spring file — cannot run in the Python venv.
 *       Included as architectural comparison alongside the Python modules.
 */

package com.example.agenticai.embabel;

import com.embabel.agent.api.annotation.Action;
import com.embabel.agent.api.annotation.Agent;
import com.embabel.agent.api.annotation.Goal;
import com.embabel.agent.api.common.OperationContext;
import com.embabel.agent.api.AgentPlatform;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.ApplicationContext;
import org.springframework.stereotype.Component;

import java.util.List;

// ---------------------------------------------------------------------------
// 1. Domain object — typed, immutable state
//    Compare: module01 uses plain Python dicts; here the schema is enforced
// ---------------------------------------------------------------------------

class TaskPlan {
    private final String description;       // original task
    private List<String> subtasks;          // after decompose
    private Integer totalStoryPoints;       // after estimate
    private boolean validated;             // after validate
    private String summary;                // after summarise — the @Goal

    public TaskPlan(String description) { this.description = description; }

    public String       getDescription()      { return description; }
    public List<String> getSubtasks()         { return subtasks; }
    public Integer      getTotalStoryPoints() { return totalStoryPoints; }
    public boolean      isValidated()         { return validated; }
    public String       getSummary()          { return summary; }

    public TaskPlan withSubtasks(List<String> s)  { this.subtasks = s;            return this; }
    public TaskPlan withStoryPoints(int pts)       { this.totalStoryPoints = pts;  return this; }
    public TaskPlan withValidated(boolean v)       { this.validated = v;           return this; }
    public TaskPlan withSummary(String s)          { this.summary = s;             return this; }
}

// ---------------------------------------------------------------------------
// 2. Services as Spring beans
//    Compare: module01 uses loose top-level functions
// ---------------------------------------------------------------------------

@Component
class DecomposerService {
    /** Ask the LLM to break a task into concrete subtasks. */
    public List<String> decompose(String description, OperationContext ctx) {
        String prompt = "Break the following software task into 3-5 concrete subtasks.\n"
                      + "Return one subtask per line, no bullet points.\n"
                      + "Task: " + description;
        String raw = ctx.getLlm().generate(prompt);
        return List.of(raw.strip().split("\\n+"));
    }
}

@Component
class EstimatorService {
    /** Ask the LLM to estimate total story points for the subtask list. */
    public int estimate(List<String> subtasks, OperationContext ctx) {
        String joined = String.join("\n", subtasks);
        String prompt = "Given these subtasks, return a single integer for the total"
                      + " story-point estimate (Fibonacci scale: 1,2,3,5,8,13,21):\n" + joined;
        String raw = ctx.getLlm().generate(prompt).strip();
        try { return Integer.parseInt(raw.replaceAll("\\D+", "")); }
        catch (NumberFormatException e) { return 8; }  // sensible default
    }
}

// ---------------------------------------------------------------------------
// 3. Embabel Agent — goal-directed, action-annotated
//
//    KEY DIFFERENCE vs Python modules:
//      module01: developer writes execution order in code (imperative)
//      LangChain: framework runs a ReAct loop (still imperative inside)
//      Embabel:  developer declares actions + guards; framework plans order
// ---------------------------------------------------------------------------

@Agent(description = "Task decomposition and estimation planning agent")
class TaskPlanningAgent {

    private final DecomposerService decomposer;
    private final EstimatorService  estimator;

    public TaskPlanningAgent(DecomposerService decomposer, EstimatorService estimator) {
        this.decomposer = decomposer;
        this.estimator  = estimator;
    }

    // ACTION 1: decompose — guard fires only when subtasks not yet populated
    @Action(description = "Decompose task into subtasks using LLM")
    public TaskPlan decompose(TaskPlan plan, OperationContext ctx) {
        if (plan.getSubtasks() != null) return plan;  // guard: already done
        List<String> subtasks = decomposer.decompose(plan.getDescription(), ctx);
        System.out.println("  [Action] decompose → " + subtasks.size() + " subtasks");
        subtasks.forEach(s -> System.out.println("    - " + s));
        return plan.withSubtasks(subtasks);
    }

    // ACTION 2: estimate — guard fires only after decompose, before estimate
    @Action(description = "Estimate total story points for the subtasks")
    public TaskPlan estimate(TaskPlan plan, OperationContext ctx) {
        if (plan.getSubtasks() == null)         return plan;  // precondition
        if (plan.getTotalStoryPoints() != null) return plan;  // guard: already done
        int pts = estimator.estimate(plan.getSubtasks(), ctx);
        System.out.println("  [Action] estimate → " + pts + " story points");
        return plan.withStoryPoints(pts);
    }

    // ACTION 3: validate — deterministic rule check (no LLM needed)
    @Action(description = "Validate plan is realistic (≤ 21 story points)")
    public TaskPlan validate(TaskPlan plan, OperationContext ctx) {
        if (plan.getTotalStoryPoints() == null) return plan;  // precondition
        if (plan.isValidated())                 return plan;  // guard: already done
        boolean ok = plan.getTotalStoryPoints() <= 21;
        System.out.println("  [Action] validate → " + (ok ? "PASS" : "WARN: may be too large"));
        return plan.withValidated(true);
    }

    // ACTION 4: summarise — fires last, uses LLM, produces the @Goal field
    @Action(description = "Write a human-readable project plan summary")
    public TaskPlan summarise(TaskPlan plan, OperationContext ctx) {
        if (!plan.isValidated())         return plan;  // precondition
        if (plan.getSummary() != null)   return plan;  // guard: already done
        String prompt = String.format(
            "Write a concise project plan summary (2-3 sentences):\n"
          + "Task: %s\nSubtasks: %s\nEstimate: %d story points\nValidated: %s",
            plan.getDescription(),
            String.join("; ", plan.getSubtasks()),
            plan.getTotalStoryPoints(),
            plan.isValidated() ? "yes" : "no"
        );
        String llmAnswer = ctx.getLlm().generate(prompt);
        System.out.println("  [Action] summarise → " + llmAnswer);
        return plan.withSummary(llmAnswer);
    }

    // GOAL: agent terminates when summary is populated
    @Goal(description = "Produce a complete task plan with estimate and summary")
    public boolean isComplete(TaskPlan plan) {
        return plan.getSummary() != null;
    }
}

// ---------------------------------------------------------------------------
// 4. Spring Boot entry point
// ---------------------------------------------------------------------------

@SpringBootApplication
public class EmbabelDemoApplication {

    public static void main(String[] args) {
        ApplicationContext ctx = SpringApplication.run(EmbabelDemoApplication.class, args);
        TaskPlanningAgent agent   = ctx.getBean(TaskPlanningAgent.class);
        AgentPlatform     platform = ctx.getBean(AgentPlatform.class);

        System.out.println("=== Embabel Task Decomposition Agent ===");
        System.out.println("Compare with: module01_raw/1.3_tool_single (same goal, different architecture)\n");

        TaskPlan plan = new TaskPlan("Build a REST API for a blog with authentication");
        System.out.println("Task: " + plan.getDescription() + "\n");

        // Embabel inspects domain state at each step and picks the right action
        // automatically — you never write the execution order in code
        TaskPlan result = (TaskPlan) platform.runAgent(agent, plan);

        System.out.println("\nFinal Plan Summary:");
        System.out.println(result.getSummary());
        System.out.println();
        System.out.println("KEY TAKEAWAY:");
        System.out.println("  module01 — you code: decompose() → estimate() → validate() → summarise()");
        System.out.println("  Embabel  — you declare @Actions with guards; framework plans the order");
        System.out.println("  Adding a new action (e.g. risk assessment) = add one @Action method");
        System.out.println("  No routing code changes needed anywhere else.");
    }
}

/*
 * COMPARISON TABLE (for student reference):
 *
 * ┌─────────────────┬────────────────────────────────────────────────────────┐
 * │ Aspect          │ Raw Python          LangChain       DSPy      Embabel │
 * ├─────────────────┼────────────────────────────────────────────────────────┤
 * │ Tool routing    │ Manual if/elif      Agent loop      Module    @Action │
 * │ Prompt mgmt     │ Hand-written        Templates       Optimized Auto    │
 * │ State tracking  │ Dicts               Memory obj      Trace     Domain  │
 * │ Planning        │ None                ReAct/Plan      Chain     Goal    │
 * │ DI / Testing    │ None                Partial         None      Spring  │
 * │ Language        │ Python              Python          Python    Java    │
 * └─────────────────┴────────────────────────────────────────────────────────┘
 */
