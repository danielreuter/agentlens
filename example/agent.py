from example.config import ls


@ls.task()
async def process_invoice(invoice: str) -> float | str:
    looks_fine = await check_integrity(invoice)

    if not looks_fine:
        return await generate_error_report(invoice)

    return await extract_total_cost(invoice)


@ls.task()
async def check_integrity(invoice: str, model: str = "gpt-4o-mini") -> bool:
    # return await ai.generate_object(
    #     model=model,
    #     type=bool,
    #     prompt=f"Return True if the invoice looks uncorrupted: {invoice.text}",
    # )
    return True


@ls.task()
async def generate_error_report(invoice: str) -> str:
    # return await ai.generate_text(
    #     model="gpt-4o",
    #     prompt=f"Write an error report for this corrupted invoice: {invoice}",
    # )
    return "shit"


@ls.task()
async def extract_total_cost(invoice: str, model: str = "gpt-4o") -> float:
    # return await ai.generate_object(
    #     model=model,
    #     type=float,
    #     prompt=f"Extract the total cost from this invoice: {invoice}",
    # )
    return 100.0
