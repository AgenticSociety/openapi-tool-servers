# 🌟 OpenAPI Tool Servers

This repository provides reference OpenAPI Tool Server implementations making it easy and secure for developers to integrate external tooling and data sources into LLM agents and workflows. Designed for maximum ease of use and minimal learning curve, these implementations utilize the widely adopted and battle-tested [OpenAPI specification](https://www.openapis.org/) as the standard protocol.

By leveraging OpenAPI, we eliminate the need for a proprietary or unfamiliar communication protocol, ensuring you can quickly and confidently build or integrate servers. This means less time spent figuring out custom interfaces and more time building powerful tools that enhance your AI applications.

## ☝️ Why OpenAPI?

- **Established Standard**: OpenAPI is a widely used, production-proven API standard backed by thousands of tools, companies, and communities.

- **No Reinventing the Wheel**: No additional documentation or proprietary spec confusion. If you build REST APIs or use OpenAPI today, you're already set.

- **Easy Integration & Hosting**: Deploy your tool servers externally or locally without vendor lock-in or complex configurations.

- **Strong Security Focus**: Built around HTTP/REST APIs, OpenAPI inherently supports widely used, secure communication methods including HTTPS and well-proven authentication standards (OAuth, JWT, API Keys).

- **Future-Friendly & Stable**: Unlike less mature or experimental protocols, OpenAPI promises reliability, stability, and long-term community support.

## 🚀 Quickstart

Get started quickly with our reference FastAPI-based implementations provided in the `servers/` directory. (You can adapt these examples into your preferred stack as needed, such as using [FastAPI](https://fastapi.tiangolo.com/), [FastOpenAPI](https://github.com/mr-fatalyst/fastopenapi) or any other OpenAPI-compatible library):

```bash
git clone https://github.com/open-webui/openapi-servers
cd openapi-servers

# Example: Installing dependencies for a specific server 'filesystem'
cd servers/filesystem
pip install -r requirements.txt
uvicorn main:app --reload
```

Now, simply point your OpenAPI-compatible clients or AI agents to your local or publicly deployed URL—no configuration headaches, no complicated transports.

## 📂 Server Examples

Reference implementations provided in this repository demonstrate common use-cases clearly and simply:

- **Filesystem Access** _(servers/filesystem)_ - Manage local file operations safely with configurable restrictions.
- **Git Server** _(servers/git)_ - Expose Git repositories for searching, reading, and possibly writing via controlled API endpoints.
- **Database Server** _(servers/database)_ - Query and inspect database schemas across common DB engines like PostgreSQL, MySQL, and SQLite.
- **Memory & Knowledge Graph** _(servers/memory)_ - Persistent memory management and semantic knowledge querying using popular and reliable storage techniques.
- **Web Search & Fetch** _(servers/web-search)_ - Retrieve and convert web-based content securely into structured API results usable by LLMs.

(More examples and reference implementations will be actively developed and continually updated.)

## 🔌 Bridge to MCP (Optional)

For your convenience, we also provide a simple, secure MCP-to-OpenAPI proxy server. This enables tool providers who initially implemented MCP servers to expose them effortlessly as standard OpenAPI-compatible APIs, ensuring existing MCP servers and resources remain accessible without additional hassle.

**Example usage:**

```bash
cd servers/mcp-proxy
pip install -r requirements.txt
uvicorn proxy:app --reload
```

This can simplify your migration or integration path, avoiding headaches typically associated with MCP's transport and security complexities.

## 📜 License

Licensed under [MIT License](LICENSE).

## 🌱 Open WebUI Community  

- For general discussions, technical exchange, and announcements, visit our [Community Discussions](https://github.com/open-webui/openapi-servers/discussions) page.
- Have ideas or feedback? Please open an issue!
