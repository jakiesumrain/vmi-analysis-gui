# Run a shared VMI Analysis server

One server on a Linux host serves Windows and Linux clients through SSH.
Each person needs an authorized SSH account or key and a VMI Analysis
account. The SSH tunnel connects the browser to the server. The VMI sign-in
selects the person's workspace.

## Set up the Linux host

1. Run the app under a dedicated Linux service account. Give that account
   read access to the shared data and write access to the workspace directory.
   Keep the shared data read-only at the filesystem level as well.
   Host only datasets from trusted sources. Keep the workspace directory
   private to the service account and the host administrator.
2. From the checkout, run `uv sync`. Run the following command once per
   VMI account to generate a password hash:

   ```sh
   uv run python -m vmi_analysis.gui.shared
   ```

3. To enable the Agent panel, install Docker or Podman for the service account,
   preferably in rootless mode, and build the bundled image from the checkout:

   ```sh
   docker build -t vmi-analysis-agent:latest packaging/agent
   ```

   Use `podman build` instead when Podman is the selected runtime, then set
   `VMI_AGENT_CONTAINER_RUNTIME=podman` in the server environment. The default
   runtime and image are `docker` and `vmi-analysis-agent:latest`; set
   `VMI_AGENT_IMAGE` to use another local image name. The container needs
   outbound network access to each user's chosen model endpoint, but no inbound
   ports. Docker group membership is host-root-equivalent; use rootless Docker
   or Podman where possible.

4. Save a JSON file outside the shared data and workspace directories.
   Replace each sample path and password hash. Keep the file readable only by
   the service account and host administrator.

   ```json
   {
     "datasets": "/srv/vmi/datasets",
     "workspaces": "/srv/vmi/workspaces",
     "max_concurrent_jobs": 2,
     "accounts": {
       "alice": {"password_hash": "<Alice's generated hash>", "admin": true},
       "bob": {"password_hash": "<Bob's generated hash>"}
     }
   }
   ```

5. Launch the server as the service account. Keep the process under your
   service manager if it must survive an administrator's SSH logout.

   ```sh
   VMI_SHARED_CONFIG=/etc/vmi/shared.json uv run vmi-analysis
   ```

Shared mode listens on `127.0.0.1:13517` by default and stays running when
all browsers disconnect. Keep the web port closed to incoming connections.
Restrict SSH to authorized users on your LAN or private VPN. For accounts that
only need the browser, ask the SSH administrator to restrict forwarding to
this port and disable shell access. See the OpenSSH
[server configuration manual](https://man.openbsd.org/sshd_config#PermitOpen).

## Connect from Linux or Windows

- **Linux:** Keep this [OpenSSH](https://man.openbsd.org/ssh#L) command running.
  Replace `ssh-user@linux-host` with your SSH username and the host name.

  ```sh
  ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:13517:127.0.0.1:13517 ssh-user@linux-host
  ```

- **Windows:** In MobaXterm, select **Tunneling → New SSH tunnel → Local port
  forwarding**. Set the forwarded port and remote port to `13517`, and the
  remote server to `127.0.0.1`. Set the SSH server, login and port for the
  Linux host. Start the tunnel and confirm that its local listener is on
  loopback.

Open [127.0.0.1:13517](http://127.0.0.1:13517) and sign in with your VMI
account. If the client port is busy, use another local port and change the
browser URL to match. Keep the remote port set to the server port.

## Use your workspace

Choose **Shared datasets** in the file picker to read experiment data. Save
calibrations, workflows and output files under **My workspace**. The server
keeps your form values across browser and server restarts. Closing the browser
or tunnel leaves your jobs running. Sign back in to see jobs from the current
server process. After a server restart, open your saved results from disk.

To save an output on your computer, open **Settings**, choose **Download
workspace file**, and select the file from **My workspace**.

The server queues work when it reaches the concurrency limit. Only VMI
administrators can change the running server's limit or stop it from Settings.
Set the startup limit in the JSON file. Restart the server to add or remove
accounts or change a password hash.

## Use the Agent panel

Every signed-in account can open **Electron → Automate → Agent**. Enter the
model endpoint's base URL, model ID, API format, and API key, then choose
**Start session**. The key is never saved in account settings or workflow files.
It is passed only to that session's container through an environment variable;
the temporary model configuration contains an environment-variable reference,
not the literal key. The key field clears after startup. **Stop session** stops
the container, removes the temporary configuration, and clears the backend's
remaining key copies. Process exit and server shutdown perform the same cleanup.

Each account has at most one agent session. It survives a browser reload or
disconnect so the user can sign back in and continue. The container can read
`/datasets` and write only `/workspace`, which map to Shared datasets and My
workspace. It runs without Linux capabilities, with a read-only root filesystem,
a 1 GiB memory limit, and one CPU. Prompts, file excerpts, command output, and
other context may be sent to the chosen model endpoint. Users should select an
endpoint they trust with the data they ask the agent to inspect. The endpoint
and model must be compatible with the selected API format.

## Security boundary

This deployment separates application accounts for an approved lab team.
Analysis jobs run under the service account. Use separate OS accounts or
isolated services if you need to run untrusted datasets or give users shell
access as the service account.
