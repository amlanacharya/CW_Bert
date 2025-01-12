# README.md

## Transaction Labeler

### Overview
The Transaction Labeler is a system designed to categorize financial transactions from bank statements into predefined income and expense categories. It leverages multiple labeling techniques, including keyword matching, regex patterns, fuzzy matching, machine learning, and BERT-based deep learning models.

### Features
- **Singleton Implementation:** Ensures a single instance of the labeler throughout the application.
- **Modular Design:** Comprises specialized labelers for maintainability and scalability.
- **Comprehensive Error Handling:** Provides detailed error messages for easier debugging.
- **Logging:** Implements standardized logging for operation tracing.
- **Data Validation:** Validates CSV inputs to ensure data integrity.
- **Performance Optimization:** Utilizes vectorized operations for efficient processing.
- **Automated Testing:** Includes unit tests to verify functionality.

### Installation
1. **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/transaction-labeler.git
    cd transaction-labeler
    ```

2. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Usage
1. **Prepare Your Bank Statement CSV:**
    Ensure your CSV file contains the required columns: `Sl. No.`, `Date`, `Description`, `Amount`, `Dr / Cr`, `Balance`.

2. **Run the Application:**
    ```bash
    python main.py --input path/to/bank_statement.csv --output path/to/output_directory/
    ```

3. **View Results:**
    The labeled transactions will be saved as `uc_income.csv` and `uc_expenses.csv` in the specified output directory.

### Configuration
You can configure input and output paths using command-line arguments or by modifying the configuration file.

### Testing
Run unit tests using `pytest`:
```bash
pytest tests/
