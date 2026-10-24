#!/usr/bin/env python3
"""A private zenmetrics workbench with the zensimx module (zensim-workbench) beside any modules the base already lists.

  assemble --base DIR --module PKG --manifest JSON --out DIR
      Copy a workbench site (a mirror of the public site with the extra-module hook, e.g. another private workbench),
      put the module beside its worker.js as pkg-zensimx/, and merge the module's manifest entry into the site's
      extra-modules.json (entries of other modules are kept; a previous zensimx entry is replaced).
  serve DIR [--host 0.0.0.0] [--port 3348] [--cert PEM --key PEM]
      Serve DIR cross-origin isolated (COOP/COEP), as the workbench needs. Browsers isolate only secure contexts, so
      other machines need HTTPS with a certificate for the host name.

The module embeds its bakes, which may be private (trained on WG1 labels): serve on the LAN only, never publish.
Exploration record: benchmarks/byconstruction_2026-10-22.md ("Serving the top candidate").
"""
import argparse
import json
import shutil
import ssl
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def assemble(a):
    base, out = Path(a.base), Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    source = json.loads((base / "site-source.json").read_text())
    shutil.copytree(base, out)
    assets = out / source.get("asset_prefix", "")
    if "extra-modules.json" not in (assets / "worker.js").read_text():
        raise SystemExit("this worker.js has no extra-module hook")
    shutil.copytree(a.module, assets / "pkg-zensimx")
    mine = json.loads(Path(a.manifest).read_text())["modules"]
    existing = assets / "extra-modules.json"
    modules = json.loads(existing.read_text())["modules"] if existing.exists() else []
    modules = [m for m in modules if m.get("name") != "zensimx"] + mine
    existing.write_text(json.dumps({"modules": modules}, indent=2) + "\n")
    print(f"{out}: modules {[m['name'] for m in modules]}")


class Isolated(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".wasm": "application/wasm", ".js": "text/javascript",
                      ".mjs": "text/javascript", ".json": "application/json"}


def serve(a):
    httpd = ThreadingHTTPServer((a.host, a.port), partial(Isolated, directory=a.dir))
    scheme = "http"
    if a.cert:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(a.cert, a.key)
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
        scheme = "https"
    print(f"serving {a.dir} on {scheme}://{a.host}:{a.port}/", flush=True)
    httpd.serve_forever()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("assemble")
    p.add_argument("--base", required=True)
    p.add_argument("--module", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("serve")
    p.add_argument("dir")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=3348)
    p.add_argument("--cert")
    p.add_argument("--key")
    a = ap.parse_args()
    {"assemble": assemble, "serve": serve}[a.cmd](a)


if __name__ == "__main__":
    main()
