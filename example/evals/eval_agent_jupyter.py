# dataset = InvoiceDataset()


# @ai.hook(check_integrity, model="o1-preview")
# def hook_check_integrity(input, output, row):
#     row.builder["is_corrupted"] = output


# @ai.hook(extract_total_cost, model="o1-preview")
# def hook_extract_total_cost(input, output, row):
#     row.builder["total_cost"] = output


# ai.run(
#     main=process_invoice,
#     dataset=dataset,
#     hooks=[hook_check_integrity, hook_extract_total_cost],
#     strict=False,  # disable strict mode to allow for partial runs
# )

# check_integrity_scores = []
# extract_total_cost_scores = []


# @ai.hook(check_integrity, model="o1-preview")
# def hook_check_integrity(input, output, row):
#     score = output == row.targets.is_corrupted
#     check_integrity_scores.append(score)


# @ai.hook(extract_total_cost, model="o1-preview")
# def hook_extract_total_cost(input, output, row):
#     score = output - row.targets.total_cost
#     extract_total_cost_scores.append(score)


# def report():
#     return f"""
#     check_integrity (% correct): {sum(check_integrity_scores) / len(check_integrity_scores)}
#     extract_total_cost (avg. error): {sum(extract_total_cost_scores) / len(extract_total_cost_scores)}
#     """


# ai.run(
#     dataset=dataset,
#     main=process_invoice,
#     hooks=[hook_check_integrity, hook_extract_total_cost],
#     report=report,
# )
