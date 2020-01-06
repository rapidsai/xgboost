/*
 Copyright (c) 2014 by Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */
package ml.dmlc.xgboost4j.java;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Optional;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

public class EnvironmentDetector {

  private static String cudaVersionFile = "/usr/local/cuda/version.txt";

  private static final Log log = LogFactory.getLog(EnvironmentDetector.class);

  public static Optional<String> getCudaVersion() {
    // firstly, try getting version from env variable
    Optional<String> version = Optional
        .ofNullable(System.getenv("RAPIDS_CUDA_VERSION"))
        .filter(literal -> literal.matches("([0-9][.0-9]*)"));
    version.ifPresent(literal -> log.info("Found CUDA version from env variable: " + literal));

    // secondly, try getting version from the "nvcc" command
    if (!version.isPresent()) {
      try {
        version = extractPattern(
            runCommand("nvcc", "--version"),
            "Cuda compilation tools, release [.0-9]+, V([.0-9]+)");
        version.ifPresent(literal -> log.info("Found CUDA version from nvcc command: " + literal));
      } catch (IOException | InterruptedException e) {
        log.debug("Could not get CUDA version with \"nvcc --version\"", e);
        version = Optional.empty();
      }
    }

    // thirdly, try reading version from CUDA home
    if (!version.isPresent()) {
      try {
        version = extractPattern(readFileContent(cudaVersionFile), "CUDA Version ([.0-9]+)");
        version.ifPresent(literal -> {
          log.info(String.format("Found CUDA version from %s: %s", cudaVersionFile, literal));
        });
      } catch (IOException e) {
        log.debug("Could not read CUDA version from CUDA home", e);
        version = Optional.empty();
      }
    }

    return version;
  }

  private static Optional<String> extractPattern(String content, String regex) {
    Matcher matcher = Pattern.compile(regex).matcher(content);
    return matcher.find() ? Optional.of(matcher.group(1)) : Optional.empty();
  }

  private static String readFileContent(String path) throws IOException {
    return new String(Files.readAllBytes(Paths.get(path)), StandardCharsets.UTF_8);
  }

  private static String runCommand(String ...command) throws IOException, InterruptedException {
    Process process = Runtime.getRuntime().exec(command);
    try (BufferedReader reader = new BufferedReader(
        new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
      return reader.lines().collect(Collectors.joining(System.lineSeparator()));
    } finally {
      assert process.waitFor() == 0;
    }
  }
}
