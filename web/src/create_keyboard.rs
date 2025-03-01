use crate::layouts::NamedDof;
use leptos::*;

#[component]
pub fn CreateKeyboard() -> impl IntoView {
    view! {
        <div class="p-2 m-2 sm:m-4 sm:p-4 rounded-lg bg-black">
            <div class="br">
                <NamedDof name="Sonne".to_owned()/>
            </div>
        </div>
    }
}
