#ifndef SQLPARSER_H
#define SQLPARSER_H

#include <string>
#include <vector>

enum class TokenType {
    SELECT,
    FROM,
    WHERE,
    COLUMN,
    LITERAL,
    OPERATOR,
    AND,
    OR,
    BETWEEN,
    UNKNOWN
};

struct Token {
    TokenType type;
    std::string value;
};

class SQLParser {
public:
    static std::vector<Token> parse(const std::string& sql);
    static std::string tokenTypeToString(TokenType type);

private:
    static TokenType getTokenType(const std::string& token);
    static bool isColumnName(const std::string& token);
    static bool isLiteral(const std::string& token);
    static void trim(std::string &s); // Helper function to trim whitespace
    static void handleParentheses(const std::string& token, std::vector<Token>& tokens); // Helper function to handle parentheses
};

#endif // SQLPARSER_H
