// Keeps manifest.json + versions.json in lockstep with package.json on a bump.
//
// Wired to the "version" npm script, so `npm version patch|minor|major`:
//   1. bumps package.json (npm does this),
//   2. runs this script -> writes the same version into manifest.json and adds
//      a versions.json entry mapping it to the current minAppVersion,
//   3. git-adds both so npm folds them into the version commit + tag.
//
// Result: one command produces a correct, taggable release; the three files can
// never drift out of sync by hand.

import { readFileSync, writeFileSync } from "node:fs";

const targetVersion = process.env.npm_package_version;
if (!targetVersion) {
    console.error("npm_package_version not set — run via `npm version`, not directly.");
    process.exit(1);
}

// manifest.json: set version; read minAppVersion to carry into versions.json.
const manifest = JSON.parse(readFileSync("manifest.json", "utf8"));
const { minAppVersion } = manifest;
manifest.version = targetVersion;
writeFileSync("manifest.json", JSON.stringify(manifest, null, 4) + "\n");

// versions.json: record the new plugin version -> min Obsidian version it needs.
const versions = JSON.parse(readFileSync("versions.json", "utf8"));
versions[targetVersion] = minAppVersion;
writeFileSync("versions.json", JSON.stringify(versions, null, 4) + "\n");

console.log(`version-bump: ${targetVersion} (minAppVersion ${minAppVersion})`);
