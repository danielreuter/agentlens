# def score(self):
#     pass

# # todo - nest this in observation
# def write_text(self, file_name: str, text: str):
#     (self._runs_dir / file_name).write_text(text)

# def write_json(self, file_name: str, data: dict):
#     pass

# def log(self, message: str) -> None:
#     run = _run_context.get()
#     if run is None:  # not in evaluation mode
#         self._log.info(message)  # fallback to regular logging
#         return

#     stack = run.observation_stack
#     if not stack:
#         raise ValueError("Observation stack unexpectedly empty")
#     current_observation = stack[-1]
#     current_observation.add_log(message)

# def write(self, name: str, content: str) -> None:
#     run = _run_context.get()
#     if run is None:  # not in evaluation mode
#         self._log.warning("Attempting to write file outside of run context")
#         return

#     stack = run.observation_stack
#     if not stack:
#         raise ValueError("Observation stack unexpectedly empty")

#     current_observation = stack[-1]
#     current_observation.add_file(name, content)

#     filepath = run.dir / name
#     filepath.write_text(content)
