import json
from plms.language_model import TransformersQG

# Initialize the model
model = TransformersQG(model='VietAI/vit5-base')

# Input text
input_text = "Chủ nghĩa duy tâm là một trong những trường phái triết học cổ xưa, khẳng định rằng thực tại chủ yếu được hình thành từ tinh thần và ý thức. Trong đó, các hình thức cơ bản của chủ nghĩa duy tâm bao gồm:\nDuy tâm khách quan: Giả định rằng có một thực tại độc lập với ý thức con người, nhưng chỉ có thể được hiểu thông qua tinh thần.\nDuy tâm chủ quan: Nhấn mạnh rằng thực tại chỉ tồn tại khi có ý thức nhận thức nó, phản ánh quan điểm của triết gia George Berkeley."


# Generate question-answer pairs
qa_pairs = model.generate_qa(input_text)

# Structure the output as a list of dictionaries with question-answer pairs
output_data = [{"question": question, "answer": answer} for question, answer in qa_pairs]

# Write the results to result.json
output_file_path = "result.json"
with open(output_file_path, "w", encoding="utf-8") as output_file:
    json.dump(output_data, output_file, ensure_ascii=False, indent=4)

print(f"Question-answer pairs have been saved to {output_file_path}")

