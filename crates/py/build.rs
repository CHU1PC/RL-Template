// macOS で cargo build するときに、拡張モジュール用のリンカ引数（-undefined dynamic_lookup）を付ける。
fn main() {
    pyo3_build_config::add_extension_module_link_args();
}
