import tabula
import json


def extract_tables_from_pdf(pdf_path):
    try:
        # Attempt to extract tables from the PDF
        tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)

        # Convert tables to JSON format
        json_data = []
        for table in tables:
            json_data.append(table.to_dict(orient='records'))

        return json_data

    except UnicodeDecodeError:
        print("Error decoding the PDF. Trying a different method...")
        # Here, you can attempt a different extraction method or just return an empty list
        return []


pdf_path = "data/table-to-pdf.pdf"
data = extract_tables_from_pdf(pdf_path)

# Save the data to a JSON file
with open('output.json', 'w', encoding='utf-8') as json_file:
    json.dump(data, json_file, indent=4)

print("Data extracted and saved to output.json")
