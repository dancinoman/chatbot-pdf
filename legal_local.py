from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
from pdfquery import PDFQuery

# Load model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Prepare dataset
def load_dataset(filepath, tokenizer, block_size=128):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=filepath,
        block_size=block_size
    )
    return dataset

# Taking pdf location
pdf_file_loc = "legal documentation/Contract_of_PurchaseSale.pdf"

# Turn PDF to text
def retrieve_pdf_text(pdf_file_loc):

    pdf_file = PDFQuery(pdf_file_loc)
    pdf_file.load()
    text_elements = pdf_file.pq('LTTextLineHorizontal')
    new_text = [t.text for t in text_elements if t.text.strip() != '' and t.text.strip().replace('_','') != ""]
    print(f'number of maximum character: {new_text}')
    print(''.join(new_text))

file_path = "legal documentation/legal.txt"
dataset = load_dataset(file_path, tokenizer)

# Set up data collator and training arguments
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
training_args = TrainingArguments(
    output_dir="./legal-text-generation",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Train model
trainer.train()

# Save model
model.save_pretrained("./legal-text-generation")
tokenizer.save_pretrained("./legal-text-generation")
