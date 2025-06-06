# algotrader
Build and back-test algorithmic trading strategies.

# 🧠 SimpleLLMAgent

An open-source project on LLM-based AI agents. 
A toy Python project demonstrating a simple LLM-based agent using OpenAI API. Includes modular structure, a basic tool (calculator) use with logging.

# 🔍 Key Features

- 🧠 **AI Agents**: Create simple LLM-based Agents.
- 📈 **Modular Design:** Definition of agents, logger, tools separated out alighed with modular design for better debugging and meintainability.
- 🖼️ **Flexibility**: Select the LLM providers you would like to use by setting up the confi file.
- ⚙️ **Extensible Design**: Add more tools and LLM providers as you please.

---

## 🗂️ Project Structure

```
algotrader/
├── algotrader.py                 # Module containing the AlgoTrader class implementing backtesting and method to display the final results  
├── strategy.py                   # Module containing the Strategy base class
├── river_stratery.py             # Module containing specific strategy, e.g. online ML algorithms from the river library
├── main.py                       # Script to run the application
├── requirements.txt              # Required Python packages
├── RILL.csv                      # Sample stock data of Reliance Industries provided for historical backtesting.

```

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/soharabhossain/algotrader.git
cd algotrader
```
### 2. Create and Activate a Virtual Environment
```bash
 python -m venv algotrader
 algotrader/Sctipts/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```
### 4. Run the Applications

  ```bash
  python main.py
  ```

---

## 📄 License

This project is open-sourced under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- Inspired by the use of Python in quantitative finance and algorithmic trading.

---

## 🤝 Contributing

Contributions are welcome! Please fork the repo and create a pull request, or open an issue to discuss what you’d like to change.

---

## 📬 Contact

For questions or suggestions, reach out to me at [soharab.hossain@gmail.com] or connect via [LinkedIn](https://www.linkedin.com/in/soharab).
