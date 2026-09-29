# Examples

This folder gives visitors a fast way to understand what OptiAgent can solve and how to phrase requests.

## Mathematical Text / LaTeX

Paste the complete model from [linear_program_sample.md](linear_program_sample.md) into the main conversation. No file upload or LLM key is required. The example has 10 variables, eight linear constraints and an independently enumerated optimum of 46.

## Facility Location

Use the built-in CSV files in `../data`:

- `facility_location_warehouses.csv`
- `facility_location_customers.csv`
- `facility_location_costs.csv`

Sample question:

```text
请根据我上传的仓库、客户和运输成本数据，求解总成本最小的仓库选址与客户分配方案，并解释为什么这样开仓。
```

## Assignment

Data file:

- [assignment_sample.json](../examples/assignment_sample.json)

Sample question:

```text
请根据这份指派数据，给出总成本最小的员工与任务分配方案。
```

## Job Shop Scheduling

Data file:

- [job_shop_scheduling_sample.json](../examples/job_shop_scheduling_sample.json)

Sample question:

```text
请根据这份作业车间调度数据，最小化 makespan，并给出每道工序的开始和结束时间。
```

## Production Mix

Data file:

- [production_mix_sample.json](../examples/production_mix_sample.json)

Sample question:

```text
请根据资源约束和产品利润，求解最优生产计划，并解释哪些资源成为瓶颈。
```
