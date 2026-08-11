 
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

class TaskPlan {
    private final String description;
    private List<String> subtasks;
    private Integer totalStoryPoints;
    private boolean validated;
    private String summary;

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

@Component
class DecomposerService {
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
    public int estimate(List<String> subtasks, OperationContext ctx) {
        String joined = String.join("\n", subtasks);
        String prompt = "Given these subtasks, return a single integer for the total"
                      + " story-point estimate (Fibonacci scale: 1,2,3,5,8,13,21):\n" + joined;
        String raw = ctx.getLlm().generate(prompt).strip();
        try { return Integer.parseInt(raw.replaceAll("\\D+", "")); }
        catch (NumberFormatException e) { return 8; }
    }
}

@Agent(description = "Task decomposition and estimation planning agent")
class TaskPlanningAgent {

    private final DecomposerService decomposer;
    private final EstimatorService  estimator;

    public TaskPlanningAgent(DecomposerService decomposer, EstimatorService estimator) {
        this.decomposer = decomposer;
        this.estimator  = estimator;
    }

    @Action(description = "Decompose task into subtasks using LLM")
    public TaskPlan decompose(TaskPlan plan, OperationContext ctx) {
        if (plan.getSubtasks() != null) return plan;
        List<String> subtasks = decomposer.decompose(plan.getDescription(), ctx);
        System.out.println("  [decompose] " + subtasks.size() + " subtasks");
        subtasks.forEach(s -> System.out.println("    - " + s));
        return plan.withSubtasks(subtasks);
    }

    @Action(description = "Estimate total story points for the subtasks")
    public TaskPlan estimate(TaskPlan plan, OperationContext ctx) {
        if (plan.getSubtasks() == null)         return plan;
        if (plan.getTotalStoryPoints() != null) return plan;
        int pts = estimator.estimate(plan.getSubtasks(), ctx);
        System.out.println("  [estimate] " + pts + " points");
        return plan.withStoryPoints(pts);
    }

    @Action(description = "Validate plan is realistic (≤ 21 story points)")
    public TaskPlan validate(TaskPlan plan, OperationContext ctx) {
        if (plan.getTotalStoryPoints() == null) return plan;
        if (plan.isValidated())                 return plan;
        boolean ok = plan.getTotalStoryPoints() <= 21;
        System.out.println("  [validate] " + (ok ? "PASS" : "WARN"));
        return plan.withValidated(true);
    }

    @Action(description = "Write a human-readable project plan summary")
    public TaskPlan summarise(TaskPlan plan, OperationContext ctx) {
        if (!plan.isValidated())         return plan;
        if (plan.getSummary() != null)   return plan;
        String prompt = String.format(
            "Write a concise project plan summary (2-3 sentences):\n"
          + "Task: %s\nSubtasks: %s\nEstimate: %d story points\nValidated: %s",
            plan.getDescription(),
            String.join("; ", plan.getSubtasks()),
            plan.getTotalStoryPoints(),
            plan.isValidated() ? "yes" : "no"
        );
        String llmAnswer = ctx.getLlm().generate(prompt);
        System.out.println("  [summarise] " + llmAnswer);
        return plan.withSummary(llmAnswer);
    }

    @Goal(description = "Produce a complete task plan with estimate and summary")
    public boolean isComplete(TaskPlan plan) {
        return plan.getSummary() != null;
    }
}

@SpringBootApplication
public class EmbabelDemoApplication {

    public static void main(String[] args) {
        ApplicationContext ctx = SpringApplication.run(EmbabelDemoApplication.class, args);
        TaskPlanningAgent agent   = ctx.getBean(TaskPlanningAgent.class);
        AgentPlatform     platform = ctx.getBean(AgentPlatform.class);

        System.out.println("=== Embabel Task Agent ===");
        System.out.println();

        TaskPlan plan = new TaskPlan("Build a REST API for a blog with authentication");
        System.out.println("Task: " + plan.getDescription() + "\n");

        TaskPlan result = (TaskPlan) platform.runAgent(agent, plan);

        System.out.println("\nFinal Plan Summary:");
        System.out.println(result.getSummary());
        System.out.println();
    }
}
