from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_restful import Resource, Api, reqparse, fields, marshal_with, abort

# -----------------------------------------------------------------------------
# App & DB Setup
# -----------------------------------------------------------------------------
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
api = Api(app)

# -----------------------------------------------------------------------------
# Database Model
# -----------------------------------------------------------------------------
class UserModel(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)

    def __repr__(self):
        return f"User(name={self.name}, email={self.email})"

# -----------------------------------------------------------------------------
# Request Parsers
# -----------------------------------------------------------------------------
user_post_args = reqparse.RequestParser()
user_post_args.add_argument(
    "name", type=str, required=True, help="User Name cannot be blank"
)
user_post_args.add_argument(
    "email", type=str, required=True, help="User Email cannot be blank"
)

user_patch_args = reqparse.RequestParser()
user_patch_args.add_argument("name", type=str)
user_patch_args.add_argument("email", type=str)

# -----------------------------------------------------------------------------
# Response Fields
# -----------------------------------------------------------------------------
user_fields = {
    "id": fields.Integer,
    "name": fields.String,
    "email": fields.String,
}

# -----------------------------------------------------------------------------
# Resources
# -----------------------------------------------------------------------------
class Users(Resource):
    @marshal_with(user_fields)
    def get(self):
        return UserModel.query.all()

    @marshal_with(user_fields)
    def post(self):
        args = user_post_args.parse_args()
        user = UserModel(name=args["name"], email=args["email"])
        db.session.add(user)
        db.session.commit()
        return user, 201


class User(Resource):
    @marshal_with(user_fields)
    def get(self, id):
        user = UserModel.query.get(id)
        if not user:
            abort(404, message="User not found")
        return user

    @marshal_with(user_fields)
    def patch(self, id):
        user = UserModel.query.get(id)
        if not user:
            abort(404, message="User not found")

        args = user_patch_args.parse_args()
        if args["name"]:
            user.name = args["name"]
        if args["email"]:
            user.email = args["email"]

        db.session.commit()
        return user

    def delete(self, id):
        user = UserModel.query.get(id)
        if not user:
            abort(404, message="User not found")

        db.session.delete(user)
        db.session.commit()
        return "", 204

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
api.add_resource(Users, "/api/users")
api.add_resource(User, "/api/users/<int:id>")

@app.route("/")
def home():
    return "<h1>Hello World!</h1>"

# -----------------------------------------------------------------------------
# Run App
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)