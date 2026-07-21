const replicaSetName = process.env.MONGO_REPLICA_SET_NAME;
const appDatabaseName = process.env.MONGO_APP_DATABASE;
const appUsername = process.env.MONGO_APP_USERNAME;
const appPassword = process.env.MONGO_APP_PASSWORD;

for (const [name, value] of Object.entries({
  MONGO_REPLICA_SET_NAME: replicaSetName,
  MONGO_APP_DATABASE: appDatabaseName,
  MONGO_APP_USERNAME: appUsername,
  MONGO_APP_PASSWORD: appPassword,
})) {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`${name} is required`);
  }
}

try {
  rs.conf();
} catch (error) {
  if (error.code !== 94 && error.codeName !== "NotYetInitialized") {
    throw error;
  }
  rs.initiate({
    _id: replicaSetName,
    members: [{ _id: 0, host: "mongo:27017" }],
  });
}

let primary = false;
for (let attempt = 0; attempt < 90; attempt += 1) {
  const hello = db.adminCommand({ hello: 1 });
  if (hello.isWritablePrimary === true) {
    primary = true;
    break;
  }
  sleep(1000);
}

if (!primary) {
  throw new Error("MongoDB replica set did not elect a primary");
}

const appDatabase = db.getSiblingDB(appDatabaseName);
// The runtime reconciles strict collection validators with collMod on every
// startup. Keep administration scoped to the application database; dbAdmin
// does not grant user-management privileges.
const roles = [
  { role: "readWrite", db: appDatabaseName },
  { role: "dbAdmin", db: appDatabaseName },
];
if (appDatabase.getUser(appUsername) === null) {
  appDatabase.createUser({
    user: appUsername,
    pwd: appPassword,
    roles,
  });
} else {
  appDatabase.updateUser(appUsername, {
    pwd: appPassword,
    roles,
  });
}

print("MongoDB replica set and application user are ready");
