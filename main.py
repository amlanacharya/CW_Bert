# main.py
import argparse
import logging
from transaction_labeler.labeler.logger_config import setup_logging
from process_bank_statement import process_bank_statement

def parse_arguments():

    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Transaction Labeler Application")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--income_output", default="uc_income.csv", help="Path to output income CSV")
    parser.add_argument("--expenses_output", default="uc_expenses.csv", help="Path to output expenses CSV")
    parser.add_argument("--log_level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    return parser.parse_args()

def main():
    """
    Main function to execute the Transaction Labeler.
    """
    args = parse_arguments()
    
   
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    setup_logging(level=log_level)
    logger = logging.getLogger(__name__)
    logger.info("Starting the Transaction Labeler application.")


    income_df, expenses_df, full_df, labeler = process_bank_statement(
        file_path=args.input,
        income_output=args.income_output,
        expenses_output=args.expenses_output
    )

    if income_df is not None and expenses_df is not None:
        logger.info("Bank statement processed successfully.")
    else:
        logger.error("Failed to process bank statement.")

if __name__ == "__main__":
    main()
