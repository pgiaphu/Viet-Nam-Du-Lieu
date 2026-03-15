# SQL_AGENT: Hệ thống AI Agent cho SQL Server 

## Tóm tắt ngắn
- Hiểu cơ sở dữ liệu từ schema và metadata.
- Trả lời câu hỏi nghiệp vụ bằng tiếng Việt hoặc tiếng Anh.
- Ưu tiên dùng kiến thức có sẵn và metadata; chỉ chạy SQL khi thực sự cần.
- Hai agent chính: dịch câu hỏi → SQL (hoặc trả lời từ kiến thức), và agent thực thi SQL an toàn.


## Danh sách cài đặt cần thiết để chạy:
1. Cài Python 3.10–3.12 trên Windows.  
2. Cài Ollama và tải model nhúng.  
3. Lấy Google AI (Gemini) API key.  
4. Cài driver ODBC cho SQL Server.  
5. Tạo file `.env` điền khóa và thông số kết nối.  
6. Cài thư viện Python.  
7. Chạy ứng dụng bằng dòng lệnh có sẵn.

---

## Bước 1: Cài Python
1. Vào https://www.python.org/downloads/ → chọn bản 3.12 (hoặc 3.11).  
2. Chạy file cài đặt, đánh dấu “Add Python to PATH”, bấm Install.  
3. Kiểm tra: mở **Command Prompt** (phím Windows, gõ `cmd`, Enter) và chạy  
   ```bash
   python --version
   ```
   Nếu thấy 3.10/3.11/3.12 là được.

## Bước 2: Cài Ollama (dùng cho nhúng văn bản)
1. Vào https://ollama.com/download → tải bản Windows, chạy cài đặt.  
2. Sau khi cài, mở Command Prompt và kiểm tra dịch vụ:  
   ```bash
   ollama --version
   ```  
3. Tải sẵn model nhúng (ví dụ `mxbai-embed-large`):  
   ```bash
   ollama run mxbai-embed-large
   ```  
   Lần đầu sẽ tự tải model, chờ hoàn tất. Dịch vụ mặc định nghe ở `http://localhost:11434`.

## Bước 3: Lấy Google AI (Gemini) API key
1. Mở trình duyệt → https://aistudio.google.com/app/apikey.  
2. Đăng nhập Google, chọn **Create API key**, đặt tên tùy ý.  
3. Copy chuỗi khóa (bắt đầu bằng `AIza...`). Giữ bí mật, không chia sẻ.

## Bước 4: Cài driver ODBC cho SQL Server
1. Tìm “ODBC Driver 17 for SQL Server” (hoặc 18) trên trang Microsoft.  
2. Tải bản phù hợp Windows, cài đặt mặc định.  
3. Sau khi cài, có thể kiểm tra nhanh trong **ODBC Data Sources** (Windows search “ODBC”).

## Bước 5: Chuẩn bị mã nguồn
1. Giải nén hoặc copy thư mục dự án này vào máy (ví dụ `D:\SQL_Agent`).  
2. Mở Command Prompt, chạy:  
   ```bash
   cd D:\SQL_Agent
   ```

## Bước 6: Tạo môi trường ảo và cài thư viện
```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```
Nếu máy chặn quyền, hãy chạy Command Prompt “Run as administrator”.

## Bước 7: Tạo file cấu hình `.env` 
Tạo file mới tên `.env` ngay trong thư mục dự án với nội dung mẫu:
```env
# Khóa LLM chính (Gemini)
GEMINI_API_KEY=<dán_api_key_google_ai>
MODEL_NAME=gemini/gemini-2.5-flash

# Dịch vụ nhúng Ollama
EMBEDDINGS_OLLAMA_BASE_URL=http://localhost:11434
EMBEDDINGS_OLLAMA_MODEL_NAME=embeddinggemma:latest

# Kết nối SQL Server
DB_DRIVER=ODBC Driver 17 for SQL Server
DB_SERVER=<ten_may_chu_or_ip>
DB_DATABASE=<ten_database>
DB_TRUSTED_CONNECTION=yes
```


## Bước 8: Chạy ứng dụng
1. Mở Command Prompt, vào thư mục dự án, kích hoạt môi trường ảo nếu chưa:  
   ```bash
   cd D:\SQL_Agent
   .venv\Scripts\activate
   ```
2. Chạy chế độ hỏi đáp tương tác:  
   ```bash
   python -m src.sql_agent_system.main
   ```
   Khi hiện prompt, gõ câu hỏi (ví dụ: “Bảng nào lưu danh mục sản phẩm?” hoặc “Doanh thu tháng 1 theo danh mục?”). Gõ `exit` để thoát.
3. Chạy một câu hỏi rồi thoát ngay:  
   ```bash
   python -m src.sql_agent_system.main "Doanh thu tháng 1 theo danh mục?"
   ```

## Bước 9: Kiểm thử nhanh sau cài đặt
- Hỏi metadata: “Danh sách bảng trong cơ sở dữ liệu?” → hệ thống trả lời mà không chạy SQL.  
- Hỏi cần dữ liệu: “TOP 10 sản phẩm bán chạy nhất?” → hệ thống sinh SQL và thực thi SELECT an toàn.  
Nếu lỗi kết nối DB, kiểm tra lại giá trị trong `.env` và quyền truy cập chỉ đọc.

## Cấu trúc thư mục chính

```
SQL_Agent/
├── README.md                           
├── pyproject.toml                      
├── .env                               
├── .gitignore                          
│
├── src/sql_agent_system/
│   ├── __init__.py
│   ├── main.py                         
│   ├── crew.py                         
│   │
│   ├── config/
│   │   ├── agents.yaml                 
│   │   └── tasks.yaml                  
│   │
│   └── tools/
│       ├── __init__.py
│       ├── db_client.py                
│       ├── database_schema_tool.py     
│       ├── sql_execution_tool.py       
│       ├── knowledge_tool.py           
│       └── __pycache__/
│
├── knowledge/
│   └── forecast_error_metrics.txt      
│       • MAPE, RMSE, MAE, WMAPE formulas
│       • Category ownership (Phú, Nhật, Mỹ, Như)
```



- `src/sql_agent_system/crew.py`: định nghĩa agent, điều kiện `needs_sql_execution`, nạp công cụ kiến thức.  
- `src/sql_agent_system/main.py`: điểm vào CLI.  
- `src/sql_agent_system/config/*.yaml`: cấu hình vai trò và nhiệm vụ của từng AI agent.  
- `src/sql_agent_system/tools/`:  
  - `database_schema_tool.py`: lấy schema, mô tả bảng/cột, mẫu dữ liệu.  
  - `knowledge_tool.py`: tìm kiếm tri thức từ `knowledge/forecast_error_metrics.txt`, nhúng Ollama, fallback từ khóa.  
  - `sql_execution_tool.py`: kiểm tra an toàn và thực thi SELECT/CTE qua SQLAlchemy + pyodbc.  
  - `db_client.py`: tạo engine từ biến môi trường.  
- `knowledge/forecast_error_metrics.txt`: công thức MAPE/RMSE/MAE/WMAPE và người phụ trách danh mục.


Minh họa một luồng chạy cơ bản với Gemma embedding:
``
knowledge/forecast_error_metrics.txt
    ↓
load_from_file()
    ↓ (Smart chunking by paragraphs)
VectorKnowledgeBase.chunks[]
    ↓ (Embed each chunk using Ollama)
VectorKnowledgeBase.embeddings[]
    ↓ (Cache embeddings + track failures)
embedding_cache{} + failed_chunks[]
    
User Query: "Who manages TRANG SỨC ECZ?"
    ↓
search(query)
    ├─ Get query embedding (with retry 3x)
    ├─ Compute cosine similarity vs all embeddings
    ├─ FALLBACK on failure → keyword search
    └─ Return top-K results sorted by relevance
        
output:
📚 FORECAST KNOWLEDGE BASE
[SOURCE 1] (Relevance: 0.85)
PHÚ: TRANG SỨC ECZ, TRANG SỨC KHÔNG GẮN ĐÁ, ...
```


## An toàn & quyền
- Chỉ cho phép SELECT/CTE; chặn DROP/DELETE/INSERT/UPDATE/ALTER/EXEC/DECLARE và ký hiệu injection.  
- Yêu cầu đặt tên bảng đầy đủ dạng `[DATA].[dbo].[TABLE]`.  
- Không lưu khóa hay chuỗi kết nối trong mã; chỉ nằm trong `.env`.  
- Tài khoản DB nên chỉ có quyền đọc.

## Gỡ lỗi thường gặp
- **Không kết nối được DB**: kiểm tra `DB_SERVER`, `DB_DATABASE`, driver ODBC, tường lửa, tài khoản chỉ đọc.  
- **Ollama lỗi hoặc quá chậm**: chắc chắn dịch vụ đang chạy (`ollama run ...`), đúng `EMBEDDINGS_OLLAMA_BASE_URL`, đã tải model nhúng.  
- **Thiếu thư viện**: kích hoạt đúng `.venv` rồi chạy lại `pip install -r requirements.txt`.  
- **Command Prompt không nhận lệnh**: đảm bảo đang ở đúng thư mục dự án và đã bật môi trường ảo.

## Mở rộng
- Thay đổi prompt hoặc luồng agent: chỉnh `config/agents.yaml` và `config/tasks.yaml`.  
- Thêm tri thức mới: thêm file vào `knowledge/`, cập nhật đường dẫn trong `crew.py` nếu đổi tên file.  
- Muốn chạy trên máy khác: chỉ cần copy mã nguồn, cài Python + Ollama, tạo `.env` mới; không copy `.env` cũ để tránh lộ khóa.

---

