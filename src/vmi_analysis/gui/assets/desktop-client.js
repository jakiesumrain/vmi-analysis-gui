const params = new URLSearchParams(location.hash.slice(1));
history.replaceState(null, "", location.pathname);
const status = document.getElementById("status");

async function signIn() {
  const response = await fetch("/client/v1/consume", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({account: params.get("account"), code: params.get("ticket")}),
  });
  if (response.status === 409) {
    status.textContent = "Sign out of the other VMI account, then open VMI again from the desktop client.";
    return;
  }
  if (!response.ok) throw new Error();
  location.replace("/");
}

signIn().catch(() => {
  status.textContent = "This launch could not be completed. Open VMI again from the desktop client.";
});
